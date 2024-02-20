
#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import pytest

import torch
from bigdl.llm.transformers import AutoModelForCausalLM, AutoModel
from transformers import LlamaTokenizer, AutoTokenizer

device = os.environ['DEVICE']
print(f'Running on {device}')

PROMPT = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"
TEST_MODEL_LIST = [
    # ("MPT-7B", AutoModelForCausalLM, AutoTokenizer, "/mnt/disk1/models/mpt-7b-chat"),  # os.environ.get('MPT_7B_ORIGIN_PATH')),
    # ("Llama2-7B", AutoModelForCausalLM, LlamaTokenizer, "/mnt/disk1/models/Llama-2-7b-chat-hf"), # os.environ.get('LLAMA2_7B_ORIGIN_PATH')),
    # ("Falcon-7B", AutoModelForCausalLM, AutoTokenizer, "/mnt/disk1/models/f"),  # os.environ.get('FALCON_7B_ORIGIN_PATH')),
    ("ChatGLM2-6B", AutoModel, AutoTokenizer, "/mnt/disk1/models/chatglm2-6b"), # os.environ.get('CHATGLM2_6B_ORIGIN_PATH')),
    # ("Mistral-7B-Instruct-v0.1", AutoModelForCausalLM, AutoTokenizer, "/mnt/disk1/models/Mistral-7B-Instruct-v0.1"),  # Need transformers==4.34.0
    # ("Baichuan-13B-Chat", AutoModelForCausalLM, AutoTokenizer, "/mnt/disk1/models/Baichuan-13B-Chat"),
    # ("Qwen-7B-Chat", AutoModelForCausalLM, AutoTokenizer, "/mnt/disk1/models/Qwen-7B-Chat"),

]

class Test_Optimize_Gpu_Model:
    def setup_method(self):

        self.layer_outputs = []
        self.pre_layer_outputs = []

    def run_optimize_gpu_model(self, Name, Model, Tokenizer, model_path, self_attn, layer_norm, lower_bound):
        with torch.inference_mode():
            def pre_forward_hook(module, input, output, layer_name):
                self.pre_layer_outputs.append(output)

            def forward_hook(module, input, output, layer_name):
                self.layer_outputs.append(output)


            tokenizer = Tokenizer.from_pretrained(model_path, trust_remote_code=True)
            input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(device)

            model = Model.from_pretrained(model_path,
                                        load_in_4bit=True,
                                        optimize_model=False,
                                        trust_remote_code=True)
            print(model)

            model = model.to(device)
            for layer_name, layer_module in model.named_modules():
                if layer_name == layer_norm:
                    layer_module.register_forward_hook(
                        lambda module, input, output, layer_name=layer_name: pre_forward_hook(module, input,
                                                                                            output, layer_name))
                if layer_name == self_attn:
                    layer_module.register_forward_hook(
                        lambda module, input, output, layer_name=layer_name: forward_hook(module, input,
                                                                                        output, layer_name))
            logits_base_model = (model(input_ids)).logits
            # the list `layer_output` has only one element.
            layer_tensor = self.layer_outputs.pop()
            model.to('cpu')

            opt_model = Model.from_pretrained(model_path,
                                            load_in_4bit=True,
                                            optimize_model=True,
                                            trust_remote_code=True)
            opt_model = opt_model.to(device)


            def replace_forward_hook(module, input, output, layer_name):
                output = self.pre_layer_outputs[0]
                return output

            for layer_name, layer_module in opt_model.named_modules():
                if layer_name == layer_norm:
                    layer_module.register_forward_hook(
                        lambda module, input, output, layer_name=layer_name: replace_forward_hook(module, input,
                                                                                                output, layer_name))
                if layer_name == self_attn:
                    layer_module.register_forward_hook(
                        lambda module, input, output, layer_name=layer_name: forward_hook(module, input,
                                                                                        output, layer_name))
            logits_optimized_model = (opt_model(input_ids)).logits
            # the list `layer_output` has only one element.
            opt_layer_tensor = self.layer_outputs[0]
            opt_model.to('cpu')

            attn_output_diff = []
            for i, (t1, t2) in enumerate(zip(layer_tensor, opt_layer_tensor)):
                if t1 is not None and t2 is not None:
                    if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
                        # 'attn_output' is of type torch.Tensor.
                        attn_output_diff.append(t1 - t2)
                    else:
                        # 'past_key_value'is of type tuple as default.
                        for i, (t3, t4) in enumerate(zip(t1, t2)):
                            if model.config.architectures[0] == "ChatGLMModel" and \
                                    hasattr(model.config, 'padded_vocab_size') and \
                                    model.config.padded_vocab_size == 65024:
                                # chatglm2's past_key_value is expanded 16x for some speedup.
                                # We need to narrow it here.
                                t4 = t4[:, :, 15:17, :]
                            attn_output_diff.append(t3 - t4)

            if 1:  # Only test use, need to be removed before commit
                output_base_dir = "./output/"
                print(output_base_dir)
                if not os.path.exists(output_base_dir):
                    os.makedirs(output_base_dir)
                
                output_model_dir = os.path.join(output_base_dir, Name)
                if not os.path.exists(output_model_dir):
                    os.makedirs(output_model_dir)
                print(output_model_dir)

                import numpy as np
                output_txt = os.path.join(output_model_dir, "RMSNorm_matrix.txt")

                layer_tensor_str = np.array2string(layer_tensor[:,:,:10].cpu().numpy(), separator=',', formatter={'float_kind': lambda x: "%.6f" % x})
                opt_layer_tensor_str = np.array2string(opt_layer_tensor[:,:,:10].cpu().numpy(), separator=',', formatter={'float_kind': lambda x: "%.6f" % x})
                
                with open(output_txt, 'a+') as file:
                    file.write("*" * 50)
                    file.write(f"\nlayer_tensor:\n{layer_tensor_str}\n\n")
                    file.write(f"opt_layer_tensor\n{opt_layer_tensor_str}\n")
                    file.write("^" * 50)

            max_diff_tensor = [torch.max(item).item() for item in attn_output_diff]
            print(max_diff_tensor)
            
            assert all(max_diff <= lower_bound for max_diff in max_diff_tensor)
    
    @pytest.mark.parametrize('Name, Model, Tokenizer, model_path',TEST_MODEL_LIST)
    def test_dynamic_functions(self, Name, Model, Tokenizer, model_path):
        if Name == "MPT-7B":
            self.MPT_7B_gpu_model(Name, Model, Tokenizer, model_path)
        elif Name == "Llama2-7B":
            self.Llama2_7B_gpu_model(Name, Model, Tokenizer, model_path)
        elif Name == "Falcon-7B":
            self.Falcon_7B_gpu_model(Name, Model, Tokenizer, model_path)
        elif Name == "ChatGLM2-6B":
            self.Chatglm2_gpu_model(Name, Model, Tokenizer, model_path)
        elif Name == "Mistral-7B-Instruct-v0.1":
            self.Mistral_gpu_model(Name, Model, Tokenizer, model_path)
        elif Name == "Baichuan-13B-Chat":
            self.Baichuan_gpu_model(Name, Model, Tokenizer, model_path)
        elif Name == "Qwen-7B-Chat":
            self.Qwen_gpu_model(Name, Model, Tokenizer, model_path)

    
    def MPT_7B_gpu_model(self, Name, Model, Tokenizer, model_path):
        # currently only need to compare the output of one self-attention layer.
        layer_norm = "transformer.blocks.31.norm_1"
        self_attn = "transformer.blocks.31.attn"
        lower_bound = 0
        self.run_optimize_gpu_model(Name, Model, Tokenizer, model_path, self_attn, layer_norm, lower_bound)

    def Llama2_7B_gpu_model(self, Name, Model, Tokenizer, model_path):
        # currently only compare the output of the last self-attention layer.
        # layer_norm = "model.layers.31.input_layernorm"
        # self_attn = "model.layers.31.self_attn"
        layer_norm = "model.layers.30.mlp"
        self_attn = "model.layers.31.input_layernorm"
        lower_bound = 21
        self.run_optimize_gpu_model(Name, Model, Tokenizer, model_path, self_attn, layer_norm, lower_bound)
    
    def Falcon_7B_gpu_model(self, Name, Model, Tokenizer, model_path):
        # currently only compare the output of the last self-attention layer.
        layer_norm = "transformer.h.31.input_layernorm"
        self_attn = "transformer.h.31.self_attention"
        lower_bound = 0
        self.run_optimize_gpu_model(Name, Model, Tokenizer, model_path, self_attn, layer_norm, lower_bound)
    
    def Chatglm2_gpu_model(self, Name, Model, Tokenizer, model_path):
        # currently only need to compare the output of one self-attention layer.
        layer_norm = "transformer.encoder.layers.26.mlp"
        self_attn = "transformer.encoder.layers.27.input_layernorm"
        lower_bound = 4e-3
        self.run_optimize_gpu_model(Name, Model, Tokenizer, model_path, self_attn, layer_norm, lower_bound)

    def Mistral_gpu_model(self, Name, Model, Tokenizer, model_path):
        # currently only need to compare the output of one self-attention layer.
        layer_norm = "model.layers.31.input_layernorm"
        self_attn = "model.layers.31.self_attn"
        lower_bound = 9e-3
        self.run_optimize_gpu_model(Name, Model, Tokenizer, model_path, self_attn, layer_norm, lower_bound)
    
    def Baichuan_gpu_model(self, Name, Model, Tokenizer, model_path):
        # currently only need to compare the output of one self-attention layer.
        layer_norm = "model.layers.39.input_layernorm"
        self_attn = "model.layers.39.self_attn"
        lower_bound = 0
        self.run_optimize_gpu_model(Name, Model, Tokenizer, model_path, self_attn, layer_norm, lower_bound)

    def Qwen_gpu_model(self, Name, Model, Tokenizer, model_path):
        # currently only need to compare the output of one self-attention layer.
        layer_norm = "transformer.h.31.ln_1"
        self_attn = "transformer.h.31.attn"
        lower_bound = 2e-3
        self.run_optimize_gpu_model(Name, Model, Tokenizer, model_path, self_attn, layer_norm, lower_bound)