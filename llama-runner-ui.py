import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
from pathlib import Path
import time
import threading

import torch
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import json


# Special token IDs
BEGIN_OF_TEXT_TOKEN_ID = 128000
END_OF_TEXT_TOKEN_ID = 128001

class AttentionHelper:
    """Helper class for attention-related operations."""
    
    @staticmethod
    def rms_norm(tensor: torch.Tensor, norm_weights: torch.Tensor, norm_eps: float) -> torch.Tensor:
        """Applies RMS normalization to the input tensor."""
        squared_mean = tensor.pow(2).mean(-1, keepdim=True)
        normalized = torch.rsqrt(squared_mean + norm_eps)
        return tensor * normalized * norm_weights

    @staticmethod
    def apply_rope(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Applies Rotary Position Embedding (RoPE) to the input tensor."""
        tensor_split_into_pairs = tensor.float().view(tensor.shape[0], -1, 2)
        d1, d2, d3 = tensor_split_into_pairs.shape
        freqs_for_each_token = torch.outer(torch.arange(d1), freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)
        tensor_as_complex_numbers = torch.view_as_complex(tensor_split_into_pairs)
        tensor_split_into_pairs_rotated = torch.view_as_real(tensor_as_complex_numbers * freqs_cis)
        return tensor_split_into_pairs_rotated.view(tensor.shape)

    @staticmethod
    def apply_masking(tensor: torch.Tensor) -> torch.Tensor:
        """Applies masking to the input tensor."""
        mask = torch.full_like(tensor, torch.finfo(tensor.dtype).min)
        mask = torch.triu(mask, diagonal=1)
        return tensor + mask
    
    
class AttentionLayer(torch.nn.Module):
    """Attention layer module."""
    
    def __init__(self, layer: int, model: dict, n_heads: int, n_kv_heads: int, dim: int, rope_freqs: torch.Tensor, norm_eps: float):
        super().__init__()
        self.layer = layer
        self.model = model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.dim = dim
        self.rope_freqs = rope_freqs
        self.norm_eps = norm_eps

    def forward(self, current_embedding: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of the attention layer."""
        # Initialize list to store QKV attentions for each head
        qkv_attention_store = []

        # Normalize the current embedding
        normalized_embedding = AttentionHelper.rms_norm(current_embedding, self.model[f"layers.{self.layer}.attention_norm.weight"], self.norm_eps)

        # Retrieve query, key, value, and output weights for the attention mechanism of the current layer
        q_layer = self.model[f"layers.{self.layer}.attention.wq.weight"]
        q_layer = q_layer.view(self.n_heads, q_layer.shape[0] // self.n_heads, self.dim)
        k_layer = self.model[f"layers.{self.layer}.attention.wk.weight"]
        k_layer = k_layer.view(self.n_kv_heads, k_layer.shape[0] // self.n_kv_heads, self.dim)
        v_layer = self.model[f"layers.{self.layer}.attention.wv.weight"]
        v_layer = v_layer.view(self.n_kv_heads, v_layer.shape[0] // self.n_kv_heads, self.dim)
        w_layer = self.model[f"layers.{self.layer}.attention.wo.weight"]

        # Iterate through each head
        for head in range(self.n_heads):
            # Extract query, key, and value weights for the current head
            q_layer_head = q_layer[head]
            k_layer_head = k_layer[head // 4]  # Key weights are shared across 4 heads
            v_layer_head = v_layer[head // 4]  # Value weights are shared across 4 heads

            # Calculate query, key, and value per token
            q_per_token = torch.matmul(normalized_embedding, q_layer_head.T)
            k_per_token = torch.matmul(normalized_embedding, k_layer_head.T)
            v_per_token = torch.matmul(normalized_embedding, v_layer_head.T)

            # Apply RoPE to query and key
            q_per_token_rotated = AttentionHelper.apply_rope(q_per_token, self.rope_freqs)
            k_per_token_rotated = AttentionHelper.apply_rope(k_per_token, self.rope_freqs)

            # Calculate query-key dot products per token
            qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T) / (128) ** 0.5

            # Apply masking to query-key dot products
            qk_per_token_masked = AttentionHelper.apply_masking(qk_per_token)

            # Apply softmax to masked query-key dot products
            qk_per_token_softmax = torch.nn.functional.softmax(qk_per_token_masked, dim=1).to(torch.bfloat16)

            # Calculate QKV attention
            qkv_attention = torch.matmul(qk_per_token_softmax, v_per_token)

            # Store QKV attention for the current head
            qkv_attention_store.append(qkv_attention)

        # Concatenate QKV attentions from all heads
        concatenated_qkv_attention = torch.cat(qkv_attention_store, dim=-1)

        # Calculate embedding delta
        embedding_delta = torch.matmul(concatenated_qkv_attention, w_layer.T)

        return embedding_delta
    
def load_tokenizer(tokenizer_path: str) -> tiktoken.Encoding:
    """Loads the tokenizer model and creates the tokenizer."""
    tokenizer_model = load_tiktoken_bpe(tokenizer_path)

    special_tokens = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|reserved_special_token_2|>",
        "<|reserved_special_token_3|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|reserved_special_token_4|>",
        "<|eot_id|>",
    ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]

    tokenize_pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

    tokenizer = tiktoken.Encoding(
        name="tokenizer.model",
        pat_str=tokenize_pattern,
        mergeable_ranks=tokenizer_model,
        special_tokens={token: len(tokenizer_model) + i for i, token in enumerate(special_tokens)},
    )

    return tokenizer

def load_model(use_mps: bool, model_path: str) -> dict:
    if use_mps:
        mps_device = torch.device("mps")
        model = torch.load(model_path, map_location=mps_device)
    else:
        model = torch.load(model_path)
    return model

def load_config(param_path: str) -> dict:
    with open(param_path, "r") as f:
        config = json.load(f)
    print(f"Configuration: {config}")
    return config

def calculate_rope_frequencies(config: dict) -> torch.Tensor:
    """Calculates RoPE (Rotary Position Embedding) frequencies."""
    rope_theta = torch.tensor(config["rope_theta"])
    zero_to_one_split_into_64_parts = torch.tensor(range(64)) / 64

    rope_scaling = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192
    }

    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]

    freqs_low = low_freq_factor / (rope_theta ** zero_to_one_split_into_64_parts[:32])
    freqs_high = high_freq_factor / (rope_theta ** zero_to_one_split_into_64_parts[32:])

    return torch.cat([freqs_low, freqs_high], dim=0)

def tokenize_prompt(tokenizer: tiktoken.Encoding, prompt: str) -> torch.Tensor:
    """Tokenizes the prompt and returns a tensor of tokens."""
    
    tokens = tokenizer.encode(prompt, allowed_special={
        '<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>', '<|begin_of_text|>'
    })
    #print(f"Encoded tokens: {tokens}")

    tokens_tensor = torch.tensor(tokens)

    prompt_split_as_tokens = [tokenizer.decode([token.item()]) for token in tokens_tensor]
    #print(''.join(prompt_split_as_tokens), end='', flush=True)

    return tokens_tensor

class LlamaGUI:
    def __init__(self, tokenizer, model, config, prompt="Prepopulated text goes here.", use_mps=False):
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.prompt = prompt
        self.use_mps = use_mps

        self.window = tk.Tk()
        self.window.title("Llama Runner")

        self.top_choices = {}
        self.dropdown_window = None
        self.is_generating = False
        self.stop_generating = False
        self.generation_thread = None

        self.right_click_time = 0
        self.play_time = 0
        self.generate_time = 0

        self.create_widgets()
        
        if self.use_mps:
            torch.set_default_device(torch.device("mps"))

    def create_widgets(self):
        self.text_editor = scrolledtext.ScrolledText(self.window, wrap=tk.WORD)
        self.text_editor.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.text_editor.insert(tk.END, self.prompt)

        self.token_editor = scrolledtext.ScrolledText(self.window, wrap=tk.WORD)
        self.token_editor.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.tokenize_button = tk.Button(self.window, text="Tokenize", command=self.tokenize_text)
        self.tokenize_button.pack(side=tk.BOTTOM)

        self.play_pause_button = tk.Button(self.window, text="Play", command=self.toggle_generation)
        self.play_pause_button.pack(side=tk.BOTTOM)
        
        # Create a frame to hold the label and entry widget
        choices_frame = tk.Frame(self.window)
        choices_frame.pack(side=tk.BOTTOM)

        # Create a label for the entry widget
        choices_label = tk.Label(choices_frame, text="Count for Step In:")
        choices_label.pack(anchor=tk.W)
        
        # Create a function to validate the input
        def validate_choices(value):
            if value.isdigit() and int(value) < 200 and int(value) > 0:
                return True
            else:
                return False
                
        # Register the validation function
        vcmd = (self.window.register(validate_choices), '%P')

        self.top_choices_entry = tk.Entry(choices_frame, width=10, validate='key', validatecommand=vcmd)
        self.top_choices_entry.insert(0, "10")
        self.top_choices_entry.pack(anchor=tk.W)

        self.step_in_button = tk.Button(self.window, text="Step In", command=self.on_step_in)
        self.step_in_button.pack(side=tk.BOTTOM)

        self.time_label = tk.Label(self.window, text="")
        self.time_label.pack(side=tk.BOTTOM)

    def on_step_in(self):
        index = self.text_editor.index(tk.INSERT)
        line_start = "1.0"
        line_end = index

        text = self.text_editor.get(line_start, line_end)
        choices = self.generate_top_choices(text)

        if self.dropdown_window:
            self.dropdown_window.destroy()

        self.show_alternate_words(choices, self.window.winfo_pointerx(), self.window.winfo_pointery())
        self.update_time_label("Step In")

    def show_alternate_words(self, choices, x, y):
        self.dropdown_window = tk.Toplevel(self.window)
        self.dropdown_window.title("Next Words")
        self.dropdown_window.geometry(f"+{x}+{y}")
        
        # Create a scrollbar
        scrollbar = tk.Scrollbar(self.dropdown_window)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        listbox = tk.Listbox(self.dropdown_window, yscrollcommand=scrollbar.set)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=listbox.yview)

        for choice, weight in choices:
            listbox.insert(tk.END, f"{choice} ({weight:.4f})")

        listbox.bind("<Double-Button-1>", lambda event: self.populate_word(event, listbox))

    def generate_top_choices(self, text):
        tokens = self.tokenizer.encode(text, allowed_special={
            '<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>', '<|begin_of_text|>'
        })
        token_embeddings = self.model["tok_embeddings.weight"][tokens].to(torch.bfloat16)

        logits = self.execute_layers(tokens, monitor_generating=False)
        top_choices = torch.topk(logits[-1], k=int(self.top_choices_entry.get()), dim=-1)
        top_choices_indices = top_choices.indices
        top_choices_weights = torch.nn.functional.softmax(top_choices.values, dim=-1)

        result = []
        for i in range(top_choices_indices.shape[0]):
            index = top_choices_indices[i].item()
            weight = top_choices_weights[i].item()
            word = self.tokenizer.decode([index])
            result.append((word, weight))

        print(f"Top {len(result)} choices: {result}")
        return result

    def populate_word(self, event, listbox):
        selection = listbox.get(listbox.curselection())
        word = selection.split(' (')[0]
        self.text_editor.insert(tk.END, word)
        self.tokenize_text()
        self.dropdown_window.destroy()

    def tokenize_text(self):
        text = self.text_editor.get("1.0", 'end-1c')
        tokens = tokenize_prompt(self.tokenizer, text)
        self.token_editor.delete("1.0", tk.END)
        self.token_editor.insert(tk.END, ' '.join(str(token.item()) for token in tokens))

    def toggle_generation(self):
        if not self.is_generating:
            self.play_time = time.time()
            self.is_generating = True
            self.play_pause_button.config(text="Pause")
            self.generation_thread = threading.Thread(target=self.generate_text)
            self.generation_thread.start()
        else:
            self.stop_generating = True
            self.generate_time = time.time() - self.play_time
            self.is_generating = False
            self.play_pause_button.config(text="Play")

    def update_time_label(self, event):
        current_time = time.time()
        if event == "Right Click":
            elapsed_time = current_time - self.right_click_time
            self.time_label.config(text=f"Right Click Time: {elapsed_time:.4f}s")
        elif event == "Play/Pause":
            elapsed_time = current_time - self.play_time
            self.time_label.config(text=f"Play/Pause Time: {elapsed_time:.4f}s")
        elif event == "Generate":
            elapsed_time = current_time - self.generate_time
            self.time_label.config(text=f"Token Time: {elapsed_time:.4f}s")

    def generate_text(self):
        if self.use_mps:
            torch.set_default_device(torch.device("mps"))
            
        self.generate_time = time.time()
        tokens_tensor = tokenize_prompt(self.tokenizer, self.text_editor.get("1.0", 'end-1c'))

        generated_token_ids = []
        token_embedding_indices = torch.cat(
            [tokens_tensor]).long()

        while self.is_generating:
            logits = self.execute_layers(token_embedding_indices, monitor_generating=True)
            if self.is_generating == False:
                break
                
            top_choice = torch.argmax(logits[-1])
            print(f"Generated token: {top_choice.item()}")
            generated_token_ids = []
            generated_token_ids.append(top_choice.item())

            if generated_token_ids[-1] in [BEGIN_OF_TEXT_TOKEN_ID, END_OF_TEXT_TOKEN_ID]:
                print("Stopping.")
                break

            token_embedding_indices = torch.cat([token_embedding_indices, torch.tensor([generated_token_ids[-1]])]).long()

            generated_text = self.tokenizer.decode(generated_token_ids)
            print('"{}"'.format(generated_text))
            self.text_editor.insert('end-1c', generated_text)
            self.text_editor.see(tk.END)
            
            self.tokenize_text()

            self.window.update()

            self.update_time_label("Generate")
            self.generate_time = time.time()

        self.stop_generating = False
        self.is_generating = False
        self.play_pause_button.config(text="Play")
        #self.update_time_label("Generate")


    def execute_layers(self, token_embedding_indices, monitor_generating=False):
        token_embeddings = self.model["tok_embeddings.weight"][token_embedding_indices].to(torch.bfloat16)
        current_embedding = token_embeddings

        for layer in range(self.config["n_layers"]):
            if monitor_generating and self.is_generating==False:
                break
            print("[*] Processing layer {}".format(layer))
            attention_layer = AttentionLayer(layer, self.model, self.config["n_heads"], self.config["n_kv_heads"],
                                             self.config["dim"], calculate_rope_frequencies(self.config), self.config["norm_eps"])
            embedding_delta = attention_layer(current_embedding)
            updated_embedding = current_embedding + embedding_delta

            normalized_updated_embedding = AttentionHelper.rms_norm(
                updated_embedding, self.model[f"layers.{layer}.ffn_norm.weight"], self.config["norm_eps"]
            )

            w1 = self.model[f"layers.{layer}.feed_forward.w1.weight"]
            w2 = self.model[f"layers.{layer}.feed_forward.w2.weight"]
            w3 = self.model[f"layers.{layer}.feed_forward.w3.weight"]

            feedforward_output = torch.matmul(
                torch.functional.F.silu(torch.matmul(normalized_updated_embedding, w1.T)) *
                torch.matmul(normalized_updated_embedding, w3.T),
                w2.T
            )

            current_embedding = updated_embedding + feedforward_output

        logits = torch.matmul(current_embedding, self.model["output.weight"].T)
        return logits

    def run(self):
        self.window.mainloop()


class LoadingWindow:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Loading Configuration")
        
        self.use_mps_var = tk.BooleanVar()
        self.model_path = tk.StringVar(value="consolidated.00.pth")
        self.tokenizer_path = tk.StringVar(value="tokenizer.model")
        self.param_path = tk.StringVar(value="params.json")
        
        self.create_widgets()
        
    def create_widgets(self):
        tk.Checkbutton(self.window, text="Use MPS", variable=self.use_mps_var).pack()
        
        tk.Label(self.window, text="Model Path:").pack()
        tk.Entry(self.window, textvariable=self.model_path).pack()
        tk.Button(self.window, text="Browse", command=self.browse_model).pack()
        
        tk.Label(self.window, text="Tokenizer Path:").pack()
        tk.Entry(self.window, textvariable=self.tokenizer_path).pack()
        tk.Button(self.window, text="Browse", command=self.browse_tokenizer).pack()
        
        tk.Label(self.window, text="Parameter Path:").pack()
        tk.Entry(self.window, textvariable=self.param_path).pack()
        tk.Button(self.window, text="Browse", command=self.browse_param).pack()
        
        tk.Button(self.window, text="Load", command=self.load_configuration).pack()
        
    def browse_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth")])
        if file_path:
            self.model_path.set(file_path)
            
    def browse_tokenizer(self):
        file_path = filedialog.askopenfilename(filetypes=[("Tokenizer Model", "*")])
        if file_path:
            self.tokenizer_path.set(file_path)
            
    def browse_param(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if file_path:
            self.param_path.set(file_path)
        
    def load_configuration(self):
        use_mps = self.use_mps_var.get()
        model_path = self.model_path.get()
        tokenizer_path = self.tokenizer_path.get()
        param_path = self.param_path.get()
        
        self.window.destroy()
        
        tokenizer = load_tokenizer(tokenizer_path)
        model = load_model(use_mps, model_path)
        config = load_config(param_path)
        
        prompt = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>

I want to parallel park car. Provide me meticulous and detailed steps.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

Sure, happy hunting. Here are detailed steps'''

        gui = LlamaGUI(tokenizer, model, config, prompt, use_mps)
        gui.run()
    
    def run(self):
        self.window.mainloop()


def main():
    loading_window = LoadingWindow()
    loading_window.run()
    
    


if __name__ == "__main__":
    main()