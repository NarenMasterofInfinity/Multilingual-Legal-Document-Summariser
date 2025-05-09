#!/usr/bin/env python3
import os
import argparse
import torch
import stanza
from sentence_transformers import SentenceTransformer
from torch import nn
import time

label_dict = {
    0: "A",   # argument
    1: "F",   # facts
    2: "P",   # precedent
    3: "R",   # ratio
    4: "RLC",# ruling by lower court
    5: "RPC",# ruling by present court
    6: "S"   # statute
}

def residual_block(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, input_dim),
        nn.ReLU(),
        nn.LayerNorm(input_dim),
        nn.Dropout(0.3)
    )
    
class LegalClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LegalClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.norm1 = nn.LayerNorm(512)
        self.residual1 = residual_block(512)
        self.fc2 = nn.Linear(512, 256)
        self.norm2 = nn.LayerNorm(256)
        self.residual2 = residual_block(256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.norm1(self.fc1(x)))
        x = self.residual1(x) + x
        x = self.relu(self.norm2(self.fc2(x)))
        x = self.residual2(x) + x
        x = self.dropout(x)
        return self.fc3(x)

def setup_argparse():
    parser = argparse.ArgumentParser(description='Process text files for legal classification.')
    parser.add_argument('--input_dir',    required=True, help='Input directory containing text files')
    parser.add_argument('--output_dir',   required=True, help='Output directory for processed files')
    parser.add_argument('--lang',         required=True, choices=['en', 'ta'], help='Language code')
    parser.add_argument('--naive',        action='store_true', default=False,
                        help='Use naive sentence splitting by period')
    return parser.parse_args()

def load_models(args, model_path):
    # Initialize Stanza pipeline for both English and Tamil
    if not args.naive:
        # Download models if not already
        stanza.download(args.lang)
        nlp = stanza.Pipeline(lang=args.lang, processors='tokenize')
    else:
        nlp = None

    # Load LaBSE embedder
    embedder = SentenceTransformer('LaBSE')
    
    # Load PyTorch model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LegalClassifier(input_dim=768, num_classes=len(label_dict)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return nlp, embedder, model, device

def split_text_naive(text):
    return [s.strip() + '.' for s in text.split('.') if s.strip()]


def process_file(filepath, nlp, embedder, model, device, output_dir, lang, naive):
    filename = os.path.basename(filepath)
    output_filepath = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Sentence splitting
    if naive:
        lines = split_text_naive(text)
    else:
        doc = nlp(text)
        # For English: doc.sentences, for Tamil: doc.sentences as well
        lines = [sent.text for sent in doc.sentences]
    
    results = []
    for line in lines:
        if not line.strip():
            continue
        embedding = embedder.encode(line, convert_to_tensor=True).to(device)
        with torch.no_grad():
            output = model(embedding.unsqueeze(0))
            prediction = torch.argmax(output, dim=1).item()
        results.append(f"{line.strip()}\t{label_dict[prediction]}")
    
    os.makedirs(output_dir, exist_ok=True)
    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write('\n&\n'.join(results))


def main():
    args = setup_argparse()
    model_path = 'best_legal_classifier_labse_q.pth'
    nlp, embedder, model, device = load_models(args, model_path)
    count = 0
    for filename in os.listdir(args.input_dir):
        filepath = os.path.join(args.input_dir, filename)
        if os.path.isfile(filepath):
            start = time.time()
            process_file(filepath, nlp, embedder, model, device,
                         args.output_dir, args.lang, args.naive)
            count += 1
            print(f"{count} files done! Time: {time.time() - start:.2f}s")

if __name__ == "__main__":
    main()
