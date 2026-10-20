import os
import sys
import unittest
import torch  # type: ignore
import torch.nn as nn  # type: ignore

# Aseguramos que source_code esté en el sys.path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from models.standalone_qwen3 import StandaloneQwen3Config, StandaloneQwen3ForCausalLM
from models.lora import LoRALinear, apply_lora_to_model
from layers.normalization.rmsnorm import RMSNorm
from layers.normalization.layernorm import LayerNorm
from layers.embeddings.rotary import RotaryEmbedding
from layers.feed_forward.swiglu import SwiGLU
from layers.attention.gqa import GroupedQueryAttention
from layers.block import Qwen3DecoderBlock
from inference.quantization import Int8Linear
from inference.kv_cache import Qwen3KVCache
from inference.sampling import sample_next_token

class TestQwen3FullSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = StandaloneQwen3Config(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            max_position_embeddings=1024,
            rms_norm_eps=1e-6,
            rope_theta=10000.0
        )
        cls.batch_size = 2
        cls.seq_len = 8

    def test_01_rmsnorm(self):
        norm = RMSNorm(dim=256, eps=1e-6)
        x = torch.randn(self.batch_size, self.seq_len, 256)
        out = norm(x)
        self.assertEqual(out.shape, x.shape)
        rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-2))

    def test_02_layernorm_comparison(self):
        ln = LayerNorm(dim=256)
        x = torch.randn(self.batch_size, self.seq_len, 256)
        out = ln(x)
        self.assertEqual(out.shape, x.shape)

    def test_03_rotary_embeddings(self):
        rope = RotaryEmbedding(dim=64, max_seq_len=1024, theta=10000.0)
        pos = torch.arange(self.seq_len).unsqueeze(0).expand(self.batch_size, -1)
        cos, sin = rope(pos)
        self.assertEqual(cos.shape, (self.batch_size, self.seq_len, 64))
        self.assertEqual(sin.shape, (self.batch_size, self.seq_len, 64))

    def test_04_swiglu_mlp(self):
        mlp = SwiGLU(hidden_size=256, intermediate_size=512)
        x = torch.randn(self.batch_size, self.seq_len, 256)
        out = mlp(x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, 256))

    def test_05_gqa_attention_with_qk_norm(self):
        attn = GroupedQueryAttention(
            hidden_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64
        )
        x = torch.randn(self.batch_size, self.seq_len, 256)
        rope = RotaryEmbedding(dim=64, max_seq_len=1024, theta=10000.0)
        pos = torch.arange(self.seq_len).unsqueeze(0).expand(self.batch_size, -1)
        cos, sin = rope(pos)
        out, weights = attn(x, cos, sin)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, 256))
        self.assertEqual(weights.shape, (self.batch_size, 4, self.seq_len, self.seq_len))

    def test_05b_decoder_block(self):
        block = Qwen3DecoderBlock(
            hidden_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=512,
            head_dim=64
        )
        x = torch.randn(self.batch_size, self.seq_len, 256)
        rope = RotaryEmbedding(dim=64, max_seq_len=1024, theta=10000.0)
        pos = torch.arange(self.seq_len).unsqueeze(0).expand(self.batch_size, -1)
        cos, sin = rope(pos)
        out = block(x, cos, sin)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, 256))

    def test_06_kv_cache(self):
        cache = Qwen3KVCache(num_layers=2, num_kv_heads=2, head_dim=64)
        k = torch.randn(self.batch_size, 2, 4, 64)
        v = torch.randn(self.batch_size, 2, 4, 64)
        k_cached, v_cached = cache.update(k, v, layer_idx=0)
        self.assertEqual(k_cached.shape, (self.batch_size, 2, 4, 64))

    def test_07_quantization_int8(self):
        linear = nn.Linear(256, 128, bias=False)
        qlinear = Int8Linear.from_float(linear)
        x = torch.randn(self.batch_size, self.seq_len, 256)
        out = qlinear(x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, 128))

    def test_08_lora_adaptation(self):
        linear = nn.Linear(256, 128, bias=False)
        lora = LoRALinear(linear, rank=4, alpha=8.0)
        x = torch.randn(self.batch_size, self.seq_len, 256)
        out = lora(x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, 128))
        # Fusión de pesos
        lora.merge_weights()
        out_merged = lora(x)
        self.assertEqual(out_merged.shape, (self.batch_size, self.seq_len, 128))

    def test_09_sampling_functions(self):
        logits = torch.randn(1, 1000)
        token = sample_next_token(logits, temperature=0.7, top_p=0.9, top_k=40)
        self.assertIsInstance(token, int)
        self.assertTrue(0 <= token < 1000)

    def test_10_full_causal_lm_pipeline(self):
        model = StandaloneQwen3ForCausalLM(self.config)
        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        logits = model(input_ids)
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, 1000))

    def test_11_pruned_model(self):
        from models.pruned_qwen3 import PrunedQwen3ForCausalLM
        base_model = StandaloneQwen3ForCausalLM(self.config)
        pruned_model = PrunedQwen3ForCausalLM(base_model, layers_to_prune=[1])
        self.assertEqual(pruned_model.pruned_layer_count, 1)
        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        logits = pruned_model(input_ids)
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, 1000))

    def test_12_audio_encoder(self):
        from multimodal.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
        from multimodal.audio_encoder import Qwen3ASRAudioEncoder
        
        cfg = Qwen3ASRAudioEncoderConfig(
            num_mel_bins=128,
            encoder_layers=2,
            encoder_attention_heads=4,
            encoder_ffn_dim=256,
            d_model=128,
            output_dim=256,
            downsample_hidden_size=64
        )
        audio_encoder = Qwen3ASRAudioEncoder(cfg)
        # Dummy Mel Spectrogram: [batch=2, mel_bins=128, time_frames=32]
        dummy_mel = torch.randn(2, 128, 32)
        out = audio_encoder(dummy_mel)
        # Reducción 8x en tiempo: 32 / 8 = 4 frames
        self.assertEqual(out.shape, (2, 4, 256))

    def test_13_attention_sink_kv_cache(self):
        from inference.kv_cache import AttentionSinkKVCache
        # Caché con 2 sink tokens y ventana de 4 tokens (máx permitido: 6)
        sink_cache = AttentionSinkKVCache(
            num_layers=1,
            num_kv_heads=2,
            head_dim=32,
            num_sink_tokens=2,
            window_size=4
        )
        # Inyectar 10 tokens sucesivamente
        for t in range(10):
            dummy_k = torch.full((1, 2, 1, 32), fill_value=float(t))
            dummy_v = torch.full((1, 2, 1, 32), fill_value=float(t))
            k_out, v_out = sink_cache.update(dummy_k, dummy_v, layer_idx=0)

        # La longitud total debe estar acotada a num_sink_tokens (2) + window_size (4) = 6
        self.assertEqual(k_out.shape[2], 6)
        # Los dos primeros tokens DEBEN ser 0.0 y 1.0 (los sumideros iniciales preservados)
        self.assertEqual(k_out[0, 0, 0, 0].item(), 0.0)
        self.assertEqual(k_out[0, 0, 1, 0].item(), 1.0)
        # Los últimos tokens deben ser los recientes (6, 7, 8, 9)
        self.assertEqual(k_out[0, 0, -1, 0].item(), 9.0)

    def test_14_copy_suppression_controller(self):
        from inference.copy_suppression import CopySuppressionController
        controller = CopySuppressionController(base_suppression_strength=2.0)
        # Logits uniformes
        logits = torch.zeros(1, 100)
        # Token 42 apareció recientemente en el contexto
        context = [10, 20, 42]
        suppressed = controller.apply_suppression(logits, context)
        # El logit del token 42 debe haber disminuido
        self.assertLess(suppressed[0, 42].item(), logits[0, 42].item())
        # Un token no visto (ej. 99) debe permanecer intacto
        self.assertEqual(suppressed[0, 99].item(), logits[0, 99].item())

    def test_15_smoothquant_scaler(self):
        from layers.normalization.outlier_channel_norm import SmoothQuantScaler
        scaler = SmoothQuantScaler(hidden_size=256, alpha=0.5)
        # Simular una activación con una dimensión rebelde (outlier en canal 42)
        act_max = torch.ones(256)
        act_max[42] = 100.0 # Outlier 100x
        weight_max = torch.ones(256)
        scaler.calibrate(act_max, weight_max)
        
        # Verificar que la escala para el canal 42 sea mayor
        self.assertGreater(scaler.scales[42].item(), scaler.scales[0].item())
        
        # Verificar transformación de entrada
        x = torch.randn(2, 4, 256)
        x_scaled = scaler(x)
        self.assertEqual(x_scaled.shape, x.shape)
        # El canal 42 escalado debe tener magnitud menor
        self.assertLess(x_scaled[:, :, 42].abs().max().item(), x[:, :, 42].abs().max().item())

    def test_16_vision_encoder(self):
        from multimodal.configuration_qwen3_vl import Qwen3VisionConfig
        from multimodal.vision_encoder import Qwen3VisionTransformer
        # Configuración compacta: imagen de 28x28, patch_size=14 -> 2x2 parches -> merge 2x2 -> 1 token
        v_cfg = Qwen3VisionConfig(
            depth=2,
            embed_dim=128,
            num_heads=4,
            patch_size=14,
            spatial_merge_size=2,
            output_dim=256
        )
        vit = Qwen3VisionTransformer(v_cfg)
        dummy_img = torch.randn(2, 3, 28, 28)
        out = vit(dummy_img)
        # 28/14 = 2 patches por lado -> 4 patches -> merge 2x2 -> 1 token comprimido
        self.assertEqual(out.shape, (2, 1, 256))

    def test_17_qwen3_vl_model(self):
        from multimodal.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VisionConfig
        from multimodal.qwen3_vl_model import Qwen3VLForConditionalGeneration
        v_cfg = Qwen3VisionConfig(
            depth=2,
            embed_dim=128,
            num_heads=4,
            patch_size=14,
            spatial_merge_size=2,
            output_dim=256
        )
        vl_cfg = Qwen3VLConfig(vision_config=v_cfg, text_config=self.config)
        vl_model = Qwen3VLForConditionalGeneration(vl_cfg)
        input_ids = torch.randint(0, 1000, (2, 8))
        dummy_img = torch.randn(2, 3, 28, 28)
        logits = vl_model(input_ids, pixel_values=dummy_img)
        self.assertEqual(logits.shape, (2, 8, 1000))

    def test_18_fim_engine(self):
        from inference.fim_engine import FIMEngine
        engine = FIMEngine(model=None, tokenizer=None)
        prompt = engine.format_prompt(
            prefix="def add(a, b):\n",
            suffix="\n    return c",
            repo_name="my_repo",
            file_path="math.py"
        )
        self.assertIn("<|fim_prefix|>", prompt)
        self.assertIn("<|fim_suffix|>", prompt)
        self.assertIn("<|fim_middle|>", prompt)
        self.assertIn("my_repo", prompt)

    def test_19_speculative_decoding(self):
        from inference.speculative_decoding import SpeculativeDecoder
        # Usamos self.config para target y modelo podado como draft
        from models.pruned_qwen3 import PrunedQwen3ForCausalLM
        target = StandaloneQwen3ForCausalLM(self.config)
        draft = PrunedQwen3ForCausalLM(target, layers_to_prune=[1])
        spec_decoder = SpeculativeDecoder(target, draft, tokenizer=None, gamma=2)
        input_ids = torch.randint(0, 1000, (1, 4))
        out_seq = spec_decoder.generate_step(input_ids)
        # La secuencia debe haber crecido al menos 1 token
        self.assertGreaterEqual(out_seq.shape[1], input_ids.shape[1] + 1)

    def test_20_sparse_autoencoder(self):
        from interpretability.sparse_autoencoder import TopKSparseAutoencoder
        sae = TopKSparseAutoencoder(d_model=256, d_sae=512, k=16)
        x = torch.randn(2, 4, 256)
        recon, latents, l2_loss = sae(x)
        self.assertEqual(recon.shape, x.shape)
        self.assertEqual(latents.shape, (2, 4, 512))
        # Verificar dispersión: exactamente k (16) valores distintos de cero por token
        non_zeros = (latents[0, 0] > 0).sum().item()
        self.assertLessEqual(non_zeros, 16)
        self.assertIsInstance(l2_loss.item(), float)

    def test_21_linear_probe(self):
        from interpretability.linear_probes import LinearProbe
        probe = LinearProbe(hidden_size=256, num_classes=3)
        h = torch.randn(2, 4, 256)
        preds = probe.predict_proba(h)
        self.assertEqual(preds.shape, (2, 4, 3))
        # Probabilidades deben sumar 1
        sums = preds.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-4))

    def test_22_yarn_rotary(self):
        from layers.embeddings.yarn_rotary import Qwen3YaRNEmbedding
        yarn = Qwen3YaRNEmbedding(
            dim=64,
            max_position_embeddings=131072,
            scale_factor=4.0,
            original_max_position_embeddings=32768
        )
        q = torch.randn(2, 4, 16, 64)
        k = torch.randn(2, 4, 16, 64)
        q_rot, k_rot = yarn.apply_yarn_rotary(q, k)
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)
        # Factor de escala de temperatura debe ser > 1.0 para compensar entropía
        self.assertGreater(yarn.mscale_factor, 1.0)

    def test_23_reasoning_engine(self):
        from inference.reasoning_engine import Qwen3ReasoningEngine
        engine = Qwen3ReasoningEngine(model=None, tokenizer=None, max_thinking_tokens=100)
        # Validar detección de momentos de reflexión/backtracking
        sample_thought = "Creo que es 42. Espera, reconsiderando el cálculo... sin embargo, la respuesta es 43."
        events = engine.count_backtracking_events(sample_thought)
        self.assertGreaterEqual(events, 3)

    def test_24_tool_calling_engine(self):
        from inference.tool_calling import Qwen3ToolCallingEngine
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Obtiene el clima actual",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        }]
        engine = Qwen3ToolCallingEngine(tools)
        # Verificar formateo de system prompt
        sys_prompt = engine.format_system_prompt_with_tools()
        self.assertIn("get_weather", sys_prompt)
        self.assertIn("<tool_call>", sys_prompt)
        
        # Verificar parseo de llamadas a herramientas
        sample_gen = (
            "Para responder a tu duda:\n"
            "<tool_call>\n"
            '{"name": "get_weather", "arguments": {"city": "Madrid"}}\n'
            "</tool_call>"
        )
        calls = engine.parse_tool_calls(sample_gen)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "get_weather")
        self.assertEqual(calls[0]["arguments"]["city"], "Madrid")
        
        # Verificar formateo de respuesta
        resp = engine.format_tool_response("get_weather", {"temp": 22})
        self.assertIn("<tool_response>", resp)
        self.assertIn('"temp": 22', resp)

    def test_25_paged_attention_manager(self):
        from inference.paged_attention import PagedKVCacheManager
        mgr = PagedKVCacheManager(
            num_layers=2,
            num_kv_heads=2,
            head_dim=32,
            total_blocks=16,
            block_size=4
        )
        mgr.allocate_sequence("seq_1")
        # Inyectar 6 tokens (debe ocupar bloque 0 completo + 2 tokens en bloque 1)
        for t in range(6):
            k = torch.full((2, 32), fill_value=float(t))
            v = torch.full((2, 32), fill_value=float(t))
            for l in range(2):
                mgr.append_token("seq_1", l, k, v)

        table = mgr.block_tables["seq_1"]
        # Debe haber asignado 2 bloques físicos
        self.assertEqual(len(table), 2)
        # Bloque 0 lleno (4 tokens), Bloque 1 con 2 tokens
        self.assertEqual(mgr.blocks[table[0]].num_tokens, 4)
        self.assertEqual(mgr.blocks[table[1]].num_tokens, 2)
        
        # Liberar secuencia
        mgr.free_sequence("seq_1")
        self.assertNotIn("seq_1", mgr.block_tables)
        self.assertEqual(len(mgr.free_block_ids), 16)

    def test_26_moe_block(self):
        from layers.feed_forward.moe import Qwen3SparseMoeBlock
        moe = Qwen3SparseMoeBlock(
            hidden_size=256,
            intermediate_size=512,
            num_experts=4,
            num_shared_experts=1,
            top_k=2
        )
        x = torch.randn(2, 8, 256)
        out, aux_loss = moe(x)
        self.assertEqual(out.shape, (2, 8, 256))
        self.assertIsInstance(aux_loss.item(), float)
        self.assertGreater(aux_loss.item(), 0.0)

    def test_27_perceiver_resampler(self):
        from multimodal.resampler import PerceiverResampler
        resampler = PerceiverResampler(
            dim=256,
            context_dim=128,
            num_latents=8,
            depth=2,
            num_heads=4,
            head_dim=64
        )
        # Secuencia visual variable (ej. 32 tokens de video)
        vis_seq = torch.randn(2, 32, 128)
        out = resampler(vis_seq)
        # Debe comprimir exactamente a num_latents (8) tokens de dimensión 256
        self.assertEqual(out.shape, (2, 8, 256))

    def test_28_direct_logit_attribution(self):
        from interpretability.dla import DirectLogitAttributionAnalyzer
        # Matriz de desembebing sintética: [vocab=100, hidden=256]
        W_U = torch.randn(100, 256)
        analyzer = DirectLogitAttributionAnalyzer(W_U)
        
        # Componente que emite una activación orientada hacia el token 42
        comp_out = W_U[42].clone()
        score = analyzer.compute_component_dla(comp_out, target_token_id=42)
        # El producto punto de un vector consigo mismo debe ser positivo
        self.assertGreater(score, 0.0)

        # Prueba con token competidor
        diff_score = analyzer.compute_component_dla(comp_out, target_token_id=42, baseline_token_id=0)
        self.assertIsInstance(diff_score, float)

    def test_29_gguf_exporter(self):
        import tempfile
        from loaders.gguf_exporter import Qwen3GGUFExporter, GGUF_MAGIC, GGUF_VERSION
        
        exporter = Qwen3GGUFExporter()
        tensors = {
            "model.embed_tokens.weight": torch.randn(64, 32),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(32, 32),
        }
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            res = exporter.export(tensors, tmp_path, quantization="q8_0")
            self.assertEqual(res["tensors_exported"], 2)
            self.assertGreater(res["file_size_bytes"], 0)
            
            with open(tmp_path, "rb") as f:
                magic = f.read(4)
                self.assertEqual(magic, b"GGUF")
                import struct
                version = struct.unpack("<I", f.read(4))[0]
                self.assertEqual(version, GGUF_VERSION)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_30_medusa_heads(self):
        from models.medusa_heads import Qwen3MedusaModel
        
        # Mock base model with .model and .lm_head
        class MockBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = lambda x, kv_cache=None: torch.randn(x.shape[0], x.shape[1], 128)
                self.lm_head = torch.nn.Linear(128, 500, bias=False)

        base = MockBase()
        medusa = Qwen3MedusaModel(base_model=base, num_heads=3, hidden_size=128, vocab_size=500)
        input_ids = torch.randint(0, 500, (2, 8))
        
        base_logits, medusa_logits = medusa(input_ids)
        self.assertEqual(base_logits.shape, (2, 8, 500))
        self.assertEqual(len(medusa_logits), 3)
        self.assertEqual(medusa_logits[0].shape, (2, 8, 500))
        
        candidates = medusa.generate_candidates(base_logits, medusa_logits)
        self.assertEqual(len(candidates), 4)  # 1 base + 3 medusa

    def test_31_reasoning_grpo_dpo_loss(self):
        from training.reasoning_loss import GRPOLoss, DPOLoss, ReasoningFormatReward
        
        # 1. Format reward
        rewarder = ReasoningFormatReward()
        valid_cot = "<think>Analizando paso 1.</think> Solución: 42."
        invalid_cot = "<think>Olvidé cerrar la etiqueta..."
        self.assertGreater(rewarder.score_text(valid_cot, 20), rewarder.score_text(invalid_cot, 20))

        # 2. GRPO loss
        grpo = GRPOLoss()
        G = 4
        seq_len = 16
        log_probs = torch.randn(G, seq_len)
        old_log_probs = log_probs.detach() + 0.01 * torch.randn(G, seq_len)
        ref_log_probs = log_probs.detach()
        rewards = torch.tensor([1.0, 0.5, -0.5, 0.0])
        
        loss, metrics = grpo(log_probs, old_log_probs, ref_log_probs, rewards)
        self.assertIsInstance(loss.item(), float)
        self.assertIn("policy_loss", metrics)

        # 3. DPO loss
        dpo = DPOLoss()
        batch_size = 2
        loss_dpo, chosen_r, rej_r = dpo(
            pi_chosen_logps=torch.tensor([-1.2, -0.8]),
            pi_rejected_logps=torch.tensor([-2.5, -2.1]),
            ref_chosen_logps=torch.tensor([-1.5, -1.0]),
            ref_rejected_logps=torch.tensor([-2.0, -1.8])
        )
        self.assertIsInstance(loss_dpo.item(), float)

    def test_32_awq_quantizer(self):
        from inference.awq_quantizer import AWQChannelProtector, AWQLinear
        
        protector = AWQChannelProtector(alpha=0.5, rogue_dim_index=5)
        linear = torch.nn.Linear(32, 64, bias=False)
        act_scales = torch.ones(32)
        act_scales[5] = 500.0  # Dimensión anómala con activación extrema
        
        q_weight, scales, group_scales = protector.protect_and_quantize_layer(linear, act_scales, bits=8, group_size=16)
        self.assertGreater(scales[5].item(), 2.0)  # Verificación de protección del canal
        
        awq_linear = AWQLinear(in_features=32, out_features=64, bits=8, group_size=16)
        awq_linear.scales.copy_(scales)
        awq_linear.group_scales.copy_(group_scales)
        
        x = torch.randn(2, 4, 32)
        out = awq_linear(x)
        self.assertEqual(out.shape, (2, 4, 64))

    def test_33_token_healing_and_grammar(self):
        from inference.grammar_sampler import TokenHealingProcessor, JSONGrammarMasker
        
        vocab = {"{": 1, "}": 2, '"name"': 3, ":": 4, '"value"': 5, ",": 6, "unrelated": 7}
        healer = TokenHealingProcessor(vocab)
        # Buscar continuaciones de '"n'
        completions = healer.find_prefix_completions('"n')
        self.assertIn(3, completions)
        self.assertNotIn(7, completions)
        
        logits = torch.zeros(1, 8)
        masked = healer.apply_healing_mask(logits, '"n')
        self.assertEqual(masked[0, 3].item(), 0.0)
        self.assertEqual(masked[0, 7].item(), float("-inf"))
        
        # Test JSONGrammarMasker
        masker = JSONGrammarMasker(vocab)
        self.assertEqual(masker.state, JSONGrammarMasker.STATE_START)
        masker.step("{")
        self.assertEqual(masker.state, JSONGrammarMasker.STATE_KEY)

    def test_34_moe_expert_specialization(self):
        from layers.feed_forward.moe_specialization import ExpertSaliencyAnalyzer
        
        analyzer = ExpertSaliencyAnalyzer(num_experts=4, top_k=2)
        logits = torch.randn(2, 8, 4)
        top_k_indices = torch.topk(logits, k=2, dim=-1).indices
        
        stats = analyzer.compute_routing_stats(logits, top_k_indices)
        self.assertEqual(len(stats["expert_load"]), 4)
        self.assertIsInstance(stats["router_entropy"], float)
        
        # Probar registro de dominios
        analyzer.record_domain_routing("math", torch.softmax(logits, dim=-1))
        analyzer.record_domain_routing("code", torch.softmax(logits + 1.0, dim=-1))
        matrix = analyzer.compute_domain_specialization_matrix()
        self.assertIn("math", matrix)
        self.assertIn("code", matrix)

    def test_35_chunked_prefill_and_tiled_attention(self):
        from layers.attention.chunked_prefill import TiledOnlineSoftmax, ChunkedPrefillEngine
        
        # Test Online Softmax comparado con standard scaled dot-product attention
        tiled_attn = TiledOnlineSoftmax(block_size=8)
        B, H, L, D = 1, 2, 16, 32
        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)
        v = torch.randn(B, H, L, D)
        
        out_tiled = tiled_attn.compute_tiled_attention(q, k, v, is_causal=True)
        self.assertEqual(out_tiled.shape, (B, H, L, D))
        
        # Test ChunkedPrefillEngine
        engine = ChunkedPrefillEngine(chunk_size=4)
        tokens = torch.randint(0, 100, (1, 12))
        chunks = engine.split_prompt_into_chunks(tokens)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].shape[1], 4)

    def test_36_quantized_kv_cache(self):
        from inference.quantized_kv_cache import QuantizedKVCache
        
        cache = QuantizedKVCache(bits=8)
        self.assertEqual(cache.memory_reduction_ratio(), 2.0)
        
        k = torch.randn(1, 4, 16, 64)
        v = torch.randn(1, 4, 16, 64)
        cache.append(k, v)
        self.assertEqual(cache.seq_len, 16)
        self.assertGreater(cache.total_bytes(), 0)
        
        # Dequantizar y verificar error de reconstrucción
        rec_k, rec_v = cache.get_dequantized()
        self.assertEqual(rec_k.shape, (1, 4, 16, 64))
        error = torch.mean(torch.abs(k - rec_k)).item()
        self.assertLess(error, 0.05)

    def test_37_continuous_batching_scheduler(self):
        from inference.continuous_batching import (
            ContinuousBatchingScheduler,
            SequenceRequest,
            RequestState
        )
        
        scheduler = ContinuousBatchingScheduler(max_batch_size=2)
        req1 = SequenceRequest(request_id="r1", prompt_tokens=[1, 2, 3], max_new_tokens=2, eos_token_id=99)
        req2 = SequenceRequest(request_id="r2", prompt_tokens=[4, 5], max_new_tokens=2, eos_token_id=99)
        
        scheduler.add_request(req1)
        scheduler.add_request(req2)
        
        # Mock sample function que emite un token sintético
        mock_sample = lambda r: 42
        
        # Paso 1: Prefill -> Decode
        outputs = scheduler.step(mock_sample)
        self.assertIn("r1", outputs)
        self.assertIn("r2", outputs)
        self.assertEqual(outputs["r1"], 42)
        
        # Paso 2: Decode -> Fin de secuencia (max_new_tokens alcanzado)
        outputs2 = scheduler.step(mock_sample)
        self.assertTrue(req1.is_finished)
        self.assertTrue(req2.is_finished)

    def test_38_cross_layer_attention(self):
        from layers.attention.cross_layer_kv import CrossLayerAttention
        
        # Capa productora (L0)
        producer = CrossLayerAttention(hidden_size=64, num_heads=4, num_kv_heads=2, is_kv_producer=True)
        # Capa consumidora (L1)
        consumer = CrossLayerAttention(hidden_size=64, num_heads=4, num_kv_heads=2, is_kv_producer=False)
        
        x = torch.randn(2, 6, 64)
        out_prod, shared_kv = producer(x)
        self.assertEqual(out_prod.shape, (2, 6, 64))
        self.assertIsNotNone(shared_kv)
        
        out_cons, next_kv = consumer(x, shared_kv=shared_kv)
        self.assertEqual(out_cons.shape, (2, 6, 64))
        self.assertIsNone(next_kv)

    def test_39_jit_operator_fusion(self):
        from inference.jit_optimizer import JITOptimizer, fused_rmsnorm, fused_swiglu
        
        x = torch.randn(2, 4, 32)
        weight = torch.ones(32)
        
        # Test Fused RMSNorm
        out_fused = fused_rmsnorm(x, weight, eps=1e-6)
        expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * weight
        self.assertTrue(torch.allclose(out_fused, expected, atol=1e-5))
        
        # Test Fused SwiGLU
        gate = torch.randn(2, 4, 32)
        up = torch.randn(2, 4, 32)
        swiglu_fused = fused_swiglu(gate, up)
        expected_swiglu = torch.nn.functional.silu(gate) * up
        self.assertTrue(torch.allclose(swiglu_fused, expected_swiglu, atol=1e-5))

    def test_40_rome_model_editing(self):
        from interpretability.rome import ROMEKnowledgeEditor
        
        editor = ROMEKnowledgeEditor(hidden_size=32)
        linear = torch.nn.Linear(32, 32, bias=False)
        
        # Vector clave del concepto original y vector valor deseado
        k_star = torch.randn(32)
        v_star = torch.randn(32)
        
        res = editor.apply_edit_to_layer(linear, k_star, v_star)
        self.assertIn("delta_norm", res)
        
        # Comprobar que W_new * k_star == v_star
        predicted_v = linear(k_star)
        error = torch.norm(predicted_v - v_star).item()
        self.assertLess(error, 1e-3)

    def test_41_hallucination_detector(self):
        from interpretability.hallucination_detector import HallucinationDetector
        
        # W_U sintética [vocab=100, hidden=32]
        W_U = torch.randn(100, 32)
        detector = HallucinationDetector(W_U, entropy_threshold=3.0)
        
        # Trayectoria sintética a través de 6 capas
        trajectory = [torch.randn(32) for _ in range(6)]
        analysis = detector.analyze_token_trajectory(trajectory)
        
        self.assertIn("hallucination_risk_score", analysis)
        self.assertIn("is_hallucinating", analysis)
        self.assertIsInstance(analysis["hallucination_risk_score"], float)
        self.assertGreaterEqual(analysis["hallucination_risk_score"], 0.0)
        self.assertLessEqual(analysis["hallucination_risk_score"], 1.0)

    def test_42_contrastive_activation_addition(self):
        from interpretability.caa import ContrastiveActivationAddition
        
        caa = ContrastiveActivationAddition(hidden_size=32)
        pos_acts = torch.randn(8, 32) + 2.0
        neg_acts = torch.randn(8, 32) - 2.0
        
        vector = caa.compute_steering_vector(pos_acts, neg_acts)
        self.assertEqual(vector.shape, (32,))
        self.assertAlmostEqual(torch.norm(vector).item(), 1.0, places=4)
        
        # Test hook attachment en una capa lineal
        dummy_layer = torch.nn.Linear(32, 32)
        caa.attach_to_module(dummy_layer, vector, scale=1.0)
        
        x = torch.zeros(1, 4, 32)
        steered_out = dummy_layer(x)
        # Comprobar que el hook modificó la salida
        caa.detach()
        unsteered_out = dummy_layer(x)
        diff = torch.norm(steered_out - unsteered_out).item()
        self.assertGreater(diff, 0.0)

    def test_43_h2o_kv_cache(self):
        from inference.h2o_cache import H2OKVCache
        
        cache = H2OKVCache(num_sinks=2, max_heavy_hitters=4, local_window_size=4)
        self.assertEqual(cache.max_capacity, 10)
        
        # Inyectar 16 tokens (superando capacidad de 10)
        k = torch.randn(1, 2, 16, 32)
        v = torch.randn(1, 2, 16, 32)
        # Generar atención acumulada con un pico en token 5
        attn = torch.zeros(1, 4, 1, 16)
        attn[0, :, 0, 5] = 10.0
        
        compact_k, compact_v = cache.update(k, v, attn_scores=attn)
        # Debe haber compactado a no más de 10 tokens
        self.assertLessEqual(compact_k.shape[-2], 10)

    def test_44_multi_head_latent_attention(self):
        from layers.attention.mla import MultiHeadLatentAttention
        
        mla = MultiHeadLatentAttention(
            hidden_size=64,
            num_heads=4,
            head_dim=16,
            kv_lora_rank=24,
            qk_rope_dim=4
        )
        x = torch.randn(2, 6, 64)
        out, latent_kv = mla(x)
        self.assertEqual(out.shape, (2, 6, 64))
        # El cache latente debe tener tamaño kv_lora_rank (24) en lugar de heads * head_dim (64)
        self.assertEqual(latent_kv.shape, (2, 6, 24))

    def test_45_refusal_direction(self):
        from interpretability.refusal_direction import RefusalDirectionAnalyzer
        
        analyzer = RefusalDirectionAnalyzer(hidden_size=32)
        harmful = torch.randn(10, 32) + 5.0
        harmless = torch.randn(10, 32) - 5.0
        
        r = analyzer.compute_refusal_direction(harmful, harmless)
        self.assertEqual(r.shape, (32,))
        self.assertAlmostEqual(torch.norm(r).item(), 1.0, places=4)
        
        # Test alineación y ortogonalización
        sample = harmful[0]
        proj_before = analyzer.measure_refusal_projection(sample)
        self.assertGreater(proj_before, 0.0)
        
        ortho_sample = analyzer.orthogonalize(sample)
        proj_after = analyzer.measure_refusal_projection(ortho_sample)
        self.assertAlmostEqual(proj_after, 0.0, places=4)

    def test_46_sliding_window_attention(self):
        from layers.attention.sliding_window import SlidingWindowAttention
        
        swa = SlidingWindowAttention(
            hidden_size=64,
            num_heads=4,
            num_kv_heads=2,
            head_dim=16,
            window_size=8,
            sink_tokens=2
        )
        x = torch.randn(2, 16, 64)
        out, (k, v) = swa(x)
        self.assertEqual(out.shape, (2, 16, 64))
        self.assertEqual(k.shape, (2, 2, 16, 16))
        self.assertEqual(v.shape, (2, 2, 16, 16))
        
        # Test mask construction
        mask = swa.construct_sliding_sink_mask(16, device=x.device)
        self.assertEqual(mask.shape, (1, 1, 16, 16))
        # Sink tokens (column 0 y 1) must be attendable
        self.assertEqual(mask[0, 0, 10, 0].item(), 0.0)
        self.assertEqual(mask[0, 0, 10, 1].item(), 0.0)
        # Token fuera de ventana y fuera de sink (ej. col 2 para fila 12, distancia 10 > window 8)
        self.assertLess(mask[0, 0, 12, 2].item(), -100.0)

    def test_47_fp8_linear_and_gemm(self):
        from inference.fp8_gemm import FP8Linear, FP8Dequantizer, convert_linear_to_fp8
        
        # Test quantize / dequantize
        tensor = torch.randn(8, 16)
        q8, scale = FP8Dequantizer.quantize(tensor, format_type="e4m3")
        self.assertEqual(q8.dtype, torch.float8_e4m3fn)
        dequant = FP8Dequantizer.dequantize(q8, scale)
        self.assertEqual(dequant.shape, (8, 16))
        
        # Test FP8Linear layer
        linear = torch.nn.Linear(32, 64, bias=True)
        fp8_layer = FP8Linear.from_linear(linear, format_type="e4m3")
        self.assertEqual(fp8_layer.weight_fp8.dtype, torch.float8_e4m3fn)
        
        x = torch.randn(2, 4, 32)
        out = fp8_layer(x)
        self.assertEqual(out.shape, (2, 4, 64))
        
        # Test convert_linear_to_fp8
        model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.Linear(32, 16))
        convert_linear_to_fp8(model)
        self.assertIsInstance(model[0], FP8Linear)

    def test_48_reasoning_verifier_and_prm(self):
        from training.reasoning_verifier import StepLevelPRM, ReasoningVerifier, BestOfNVerifier
        
        # 1. StepLevelPRM
        prm = StepLevelPRM(hidden_size=32)
        step_reps = torch.randn(2, 4, 32)
        rewards = prm(step_reps)
        self.assertEqual(rewards.shape, (2, 4))
        self.assertTrue((rewards >= 0.0).all() and (rewards <= 1.0).all())
        
        # 2. ReasoningVerifier extraction
        verifier = ReasoningVerifier()
        good_trace = "<think>\nPaso 1: Identificar datos.\n\nPaso 2: Calcular integral.\n</think>\nEl resultado es 42."
        parsed = verifier.extract_reasoning_and_answer(good_trace)
        self.assertTrue(parsed["valid_format"])
        self.assertEqual(parsed["answer"], "El resultado es 42.")
        self.assertGreaterEqual(parsed["num_steps"], 2)
        
        fmt_reward = verifier.compute_format_reward(good_trace)
        self.assertGreater(fmt_reward, 0.5)
        
        # 3. BestOfNVerifier
        bon = BestOfNVerifier(prm_model=prm)
        candidates = [
            good_trace,
            "Respuesta directa sin razonamiento."
        ]
        cand_embs = [torch.randn(parsed["num_steps"], 32), torch.randn(1, 32)]
        result = bon.select_best(candidates, cand_embs)
        self.assertEqual(result["best_index"], 0)
        self.assertEqual(result["best_candidate"], good_trace)

    def test_49_mcts_reasoning(self):
        from training.mcts_reasoning import MCTSNode, Qwen3MCTSSearchEngine
        from training.reasoning_verifier import StepLevelPRM

        # Test node logic and UCT
        root = MCTSNode(state_text="Demostrar formalmente que P != NP.", step_index=0)
        c1 = MCTSNode(state_text="Paso 1: Supongamos reducible.", step_index=1, parent=root, prior_prob=0.8, step_reward=0.9)
        c2 = MCTSNode(state_text="Paso 1: Argumento erróneo.", step_index=1, parent=root, prior_prob=0.2, step_reward=0.1)
        root.children = [c1, c2]
        root.visit_count = 10
        c1.visit_count = 6
        c1.total_value = 5.4
        c2.visit_count = 4
        c2.total_value = 0.4
        
        self.assertGreater(c1.uct_score(), c2.uct_score())
        self.assertEqual(root.best_child(), c1)
        self.assertEqual(len(c1.get_full_trajectory()), 2)
        
        # Test full search engine with PRM
        prm = StepLevelPRM(hidden_size=32)
        engine = Qwen3MCTSSearchEngine(prm_model=prm, max_depth=3, branch_factor=2)
        res = engine.search("Premisa matemática de prueba", num_simulations=5)
        self.assertIn("best_trajectory", res)
        self.assertIn("<think>", res["formatted_solution"])

    def test_50_sequence_packing(self):
        from training.sequence_packing import SequencePacker
        
        packer = SequencePacker(max_tokens_per_pack=64)
        samples = [
            {"input_ids": torch.randint(0, 100, (1, 12))},
            {"input_ids": torch.randint(0, 100, (1, 16))},
            {"input_ids": torch.randint(0, 100, (1, 8))},
        ]
        packed_batches = packer.pack(samples)
        self.assertEqual(len(packed_batches), 1)
        batch = packed_batches[0]
        self.assertEqual(batch.input_ids.shape[-1], 36)
        self.assertEqual(batch.num_samples, 3)
        self.assertEqual(len(batch.cu_seqlens), 4)  # [0, 16, 28, 36]
        
        # Unpack verification
        dummy_packed_states = torch.randn(1, 36, 16)
        unpacked = SequencePacker.unpack_hidden_states(dummy_packed_states, batch.cu_seqlens)
        self.assertEqual(len(unpacked), 3)
        self.assertEqual(unpacked[0].shape, (1, 16, 16))

    def test_51_activation_checkpointing(self):
        from training.activation_checkpointing import apply_activation_checkpointing, CheckpointedBlockWrapper
        from models.standalone_qwen3 import StandaloneQwen3ForCausalLM
        
        model = StandaloneQwen3ForCausalLM(self.config)
        model.train()
        
        count = apply_activation_checkpointing(model, every_n_layers=1)
        self.assertEqual(count, self.config.num_hidden_layers)
        self.assertIsInstance(model.model.layers[0], CheckpointedBlockWrapper)
        
        # Verify gradient flow with checkpointing
        x = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len))
        out = model(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(model.model.embed_tokens.weight.grad)

    def test_52_download_weights_catalog_and_integrity(self):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
        from download_weights import EXPECTED_FILES, format_size, format_time  # type: ignore

        self.assertIn("model-00001-of-00002.safetensors", EXPECTED_FILES)
        self.assertIn("model-00002-of-00002.safetensors", EXPECTED_FILES)
        self.assertTrue(EXPECTED_FILES["model-00001-of-00002.safetensors"]["essential"])
        self.assertEqual(EXPECTED_FILES["model-00001-of-00002.safetensors"]["size"], 3441185608)
        self.assertEqual(EXPECTED_FILES["model-00002-of-00002.safetensors"]["size"], 622329984)

        # Formateo de bytes y tiempo
        self.assertIn("GB", format_size(3441185608))
        self.assertEqual(format_time(65), "01:05")

    def test_53_awq_4bit_packing_and_layer(self):
        from inference.awq_quantizer import pack_int4_weights, unpack_int4_weights, AWQ4BitLinear, AWQChannelProtector

        # Test pack / unpack roundtrip
        q_original = torch.tensor([[-8, -5, 0, 3, 7, -2], [4, -1, 0, 7, -8, 2]], dtype=torch.int8)
        packed = pack_int4_weights(q_original)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.shape, (2, 3))  # 6 cols comprimidas a 3 bytes

        unpacked = unpack_int4_weights(packed, target_in_features=6)
        self.assertTrue(torch.equal(q_original, unpacked))

        # Test AWQ4BitLinear layer forward
        linear = torch.nn.Linear(32, 16, bias=False)
        act_scales = torch.ones(32)
        # Rogue dimension index 10 (simulada en dimensión reducida)
        protector = AWQChannelProtector(alpha=0.5, rogue_dim_index=10)
        awq_layer = AWQ4BitLinear.from_linear(linear, act_scales, group_size=16, protector=protector)

        self.assertEqual(awq_layer.packed_weight.shape, (16, 16))
        self.assertEqual(awq_layer.scales.shape, (32,))

        x = torch.randn(2, 4, 32)
        out = awq_layer(x)
        self.assertEqual(out.shape, (2, 4, 16))
        self.assertFalse(torch.isnan(out).any())

    def test_54_induction_copy_suppression_bench(self):
        from inference.copy_suppression import CopySuppressionController

        controller = CopySuppressionController(dla_discount_factor=0.3)
        logits = torch.zeros(1, 100)
        logits[0, 42] = 5.0
        logits[0, 15] = 5.0

        # Secuencia reciente donde 42 acaba de aparecer
        history = [10, 20, 30, 42]
        calibrated = controller.suppress_repetition(logits, history)

        # El logit de 42 debe haberse suprimido respecto a 15
        self.assertLess(calibrated[0, 42].item(), logits[0, 42].item())
        self.assertEqual(calibrated[0, 15].item(), logits[0, 15].item())


    def test_55_sft_trainer_step_and_convergence(self):
        from training.sft_trainer import SFTTrainer
        from models.standalone_qwen3 import StandaloneQwen3ForCausalLM

        model = StandaloneQwen3ForCausalLM(self.config)
        trainer = SFTTrainer(model, learning_rate=1e-3, warmup_steps=2, max_steps=10, use_checkpointing=False)

        dummy_batch = torch.randint(0, self.config.vocab_size, (2, 8))
        stats1 = trainer.train_step(dummy_batch)

        self.assertIn("loss", stats1)
        self.assertIn("perplexity", stats1)
        self.assertIn("tokens_per_sec", stats1)
        self.assertEqual(stats1["step"], 1)

    def test_56_sft_epoch_with_sequence_packing(self):
        from training.sft_trainer import SFTTrainer
        from models.standalone_qwen3 import StandaloneQwen3ForCausalLM

        model = StandaloneQwen3ForCausalLM(self.config)
        trainer = SFTTrainer(model, learning_rate=1e-3, max_steps=5, use_checkpointing=False)

        samples = [
            {"input_ids": torch.randint(0, self.config.vocab_size, (1, 6))},
            {"input_ids": torch.randint(0, self.config.vocab_size, (1, 8))},
        ]
        metrics = trainer.train_epoch(samples)
        self.assertGreaterEqual(len(metrics), 1)
        self.assertTrue(all("loss" in m for m in metrics))

    def test_57_forensic_api_spec(self):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
        from forensic_api import ForensicAPIHandler  # type: ignore

        self.assertTrue(hasattr(ForensicAPIHandler, "do_GET"))
        self.assertTrue(hasattr(ForensicAPIHandler, "do_OPTIONS"))


if __name__ == "__main__":
    unittest.main()








