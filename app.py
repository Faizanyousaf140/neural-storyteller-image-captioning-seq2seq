import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from collections import Counter

# ============================================================
# Configuration
# ============================================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

# Model hyperparameters (must match training configuration)
HIDDEN_SIZE = 512
EMBED_SIZE = 256
INPUT_SIZE = 2048  # ResNet-50 feature dimension


# ============================================================
# Model Definitions
# ============================================================
class Vocabulary:
    """Vocabulary class for mapping between words and indices.
    Must match the training notebook definition exactly for torch.load to work."""
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.freqs = Counter()

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, captions_list):
        for captions in captions_list:
            for caption in captions:
                self.freqs.update(caption.split())
        idx = 4
        for word, freq in self.freqs.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokens = text.split()
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokens]


class Encoder(nn.Module):
    """Image encoder using a linear projection from ResNet features."""
    def __init__(self, input_size=2048, hidden_size=512):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, features):
        return self.relu(self.linear(features))


class Decoder(nn.Module):
    """Caption decoder using LSTM."""
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions[:, :-1])
        h0 = features.unsqueeze(0)
        c0 = torch.zeros_like(h0)
        lstm_out, _ = self.lstm(embeddings, (h0, c0))
        return self.linear(lstm_out)


class Seq2Seq(nn.Module):
    """Complete Seq2Seq model combining Encoder and Decoder."""
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, features, captions):
        encoder_outputs = self.encoder(features)
        return self.decoder(encoder_outputs, captions)


# ============================================================
# Inference Functions
# ============================================================
def greedy_search(model, features, vocab, device, max_len=20):
    """Generate caption using greedy search."""
    model.eval()
    with torch.no_grad():
        encoder_out = model.encoder(features.unsqueeze(0))
        current_token = torch.tensor([vocab.stoi["<start>"]], device=device)
        caption_tokens = [current_token.item()]
        hidden = encoder_out.unsqueeze(0)
        cell = torch.zeros_like(hidden)

        for _ in range(max_len):
            word_embed = model.decoder.embedding(current_token.unsqueeze(0))
            lstm_out, (hidden, cell) = model.decoder.lstm(word_embed, (hidden, cell))
            output = model.decoder.linear(lstm_out.squeeze(0))
            current_token = output.argmax(1)
            token_id = current_token.item()
            caption_tokens.append(token_id)
            if token_id == vocab.stoi["<end>"] or token_id == vocab.stoi["<pad>"]:
                break

        words = []
        for token_id in caption_tokens[1:]:
            if token_id in [vocab.stoi["<end>"], vocab.stoi["<pad>"]]:
                break
            words.append(vocab.itos.get(token_id, "<unk>"))
        return ' '.join(words)


def beam_search(model, features, vocab, device, beam_width=3, max_len=20):
    """Generate caption using beam search with proper state cloning."""
    model.eval()
    with torch.no_grad():
        encoder_out = model.encoder(features.unsqueeze(0))
        beams = [(
            [vocab.stoi["<start>"]],
            0.0,
            encoder_out.unsqueeze(0).clone(),
            torch.zeros_like(encoder_out.unsqueeze(0)).clone()
        )]
        completed_beams = []

        for step in range(max_len):
            candidates = []
            for seq, log_prob, hidden, cell in beams:
                last_token = seq[-1]
                if last_token == vocab.stoi["<end>"] or last_token == vocab.stoi["<pad>"]:
                    completed_beams.append((seq, log_prob))
                    continue

                token_tensor = torch.tensor([last_token], device=device)
                embed = model.decoder.embedding(token_tensor).unsqueeze(0)
                lstm_out, (new_hidden, new_cell) = model.decoder.lstm(embed, (hidden, cell))
                logits = model.decoder.linear(lstm_out.squeeze(0))
                log_probs = torch.log_softmax(logits, dim=1).squeeze(0)
                topk_log_probs, topk_indices = log_probs.topk(beam_width)

                for i in range(beam_width):
                    token_id = topk_indices[i].item()
                    token_log_prob = topk_log_probs[i].item()
                    new_seq = seq + [token_id]
                    new_log_prob = log_prob + token_log_prob
                    candidates.append((
                        new_seq, new_log_prob,
                        new_hidden.clone(), new_cell.clone()
                    ))

            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]

        for seq, log_prob, _, _ in beams:
            if seq[-1] != vocab.stoi["<end>"]:
                seq.append(vocab.stoi["<end>"])
            completed_beams.append((seq, log_prob))

        if not completed_beams:
            return "a picture"

        best_seq = max(completed_beams, key=lambda x: x[1])[0]
        words = []
        for token_id in best_seq[1:]:
            if token_id == vocab.stoi["<end>"] or token_id == vocab.stoi["<pad>"]:
                break
            words.append(vocab.itos.get(token_id, "<unk>"))

        caption = ' '.join(words)
        return caption if caption.strip() else "a picture"


# ============================================================
# Feature Extraction
# ============================================================
@st.cache_resource
def load_resnet_encoder():
    """Load pre-trained ResNet-50 as feature extractor."""
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()
    return resnet


@st.cache_resource
def load_model():
    """Load the trained Seq2Seq model and vocabulary."""
    if not os.path.exists(MODEL_PATH):
        return None, None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    vocab = checkpoint['vocab']

    # Build model
    vocab_size = len(vocab)
    encoder = Encoder(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
    decoder = Decoder(embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE,
                      vocab_size=vocab_size, num_layers=1)
    model = Seq2Seq(encoder, decoder).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, vocab, device


def extract_features(image, resnet, device):
    """Extract features from an image using ResNet-50."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet(img_tensor).view(1, -1)
    return features.squeeze(0)


# ============================================================
# Streamlit App
# ============================================================
def main():
    st.set_page_config(
        page_title="🖼️ Image Captioning",
        page_icon="🖼️",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
        }
        .caption-box {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            border-left: 4px solid #4CAF50;
        }
        .caption-label {
            font-weight: bold;
            color: #333;
            font-size: 1.1rem;
        }
        .caption-text {
            font-size: 1.2rem;
            color: #1a1a2e;
            margin-top: 0.5rem;
        }
        .info-box {
            background-color: #e8f4f8;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='main-header'>🖼️ Image Captioning with Deep Learning</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Upload an image and get an AI-generated caption using a ResNet-50 + LSTM model trained on Flickr30k</p>", unsafe_allow_html=True)

    st.divider()

    # Check model
    if not os.path.exists(MODEL_PATH):
        st.error(f"⚠️ Model file not found! Please place `best_model.pth` in: `{MODEL_DIR}`")
        st.stop()

    # Load model
    with st.spinner("Loading model..."):
        model, vocab, device = load_model()
        resnet = load_resnet_encoder()
        resnet = resnet.to(device)

    if model is None:
        st.error("Failed to load the model. Please check the model file.")
        st.stop()

    st.success(f"✅ Model loaded successfully! (Device: {device}, Vocab size: {len(vocab)})")

    # UI Layout: move upload & settings to the sidebar, main area for image + captions
    st.sidebar.header("📤 Upload & Settings")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "webp"])

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Decoding Settings")
    method = st.sidebar.radio("Decoding Method:", ["Both", "Greedy Search", "Beam Search (k=3)"], index=0)

    beam_width = 3
    if "Beam" in method or method == "Both":
        beam_width = st.sidebar.slider("Beam Width:", 2, 10, 3)
    max_len = st.sidebar.slider("Max Caption Length:", 10, 40, 20)

    # Main content area: image (left) and captions (right)
    col_img, col_caption = st.columns([1, 1.1])

    with col_img:
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.subheader("🖼️ Uploaded Image")
            st.image(image, use_container_width=True)
        else:
            st.markdown("""
            <div class='info-box'>
                <h3>👈 Upload an image to get started!</h3>
                <p>Supported formats: JPG, JPEG, PNG, BMP, WEBP</p>
                <br>
                <h4>How it works:</h4>
                <ol>
                    <li><strong>Feature Extraction</strong>: ResNet-50 extracts visual features</li>
                    <li><strong>Caption Generation</strong>: LSTM decoder generates caption word by word</li>
                    <li><strong>Decoding</strong>: Greedy Search or Beam Search</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)

    with col_caption:
        if uploaded_file:
            with st.spinner("Generating caption..."):
                features = extract_features(image, resnet, device)
                greedy_cap = ""
                beam_cap = ""
                if method in ["Greedy Search", "Both"]:
                    greedy_cap = greedy_search(model, features, vocab, device, max_len)
                if method in ["Beam Search (k=3)", "Both"]:
                    beam_cap = beam_search(model, features, vocab, device, beam_width, max_len)

            st.subheader("📝 Generated Captions")

            if greedy_cap:
                st.markdown(f"""
                <div class='caption-box'>
                    <div class='caption-label'>🔍 Greedy Search:</div>
                    <div class='caption-text'>{greedy_cap}</div>
                </div>
                """, unsafe_allow_html=True)
                st.download_button("⬇️ Download Greedy Caption", greedy_cap, file_name="greedy_caption.txt")

            if beam_cap:
                st.markdown(f"""
                <div class='caption-box' style='border-left-color: #2196F3;'>
                    <div class='caption-label'>🎯 Beam Search (k={beam_width}):</div>
                    <div class='caption-text'>{beam_cap}</div>
                </div>
                """, unsafe_allow_html=True)
                st.download_button("⬇️ Download Beam Caption", beam_cap, file_name="beam_caption.txt")

    st.divider()
    st.markdown("<p style='text-align: center; color: gray; font-size: 0.9rem;'>🧠 Model: ResNet-50 + LSTM | 📚 Trained on Flickr30k</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
