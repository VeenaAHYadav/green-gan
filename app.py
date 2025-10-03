import os
import sys
import streamlit as st
import torch
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.green_gan import GreenGenerator, GreenDiscriminator
from scripts.utils import load_model

LATENT_DIM = 16
N_FEATURES = 78  
DEVICE = "cpu"  


@st.cache_resource  
def load_models():
    """Load generator and discriminator models."""
    # Instantiate models
    gen = GreenGenerator(latent_dim=LATENT_DIM, n_features=N_FEATURES)
    disc = GreenDiscriminator(input_dim=N_FEATURES)

    gen.to(DEVICE)
    disc.to(DEVICE)
    gen.eval()
    disc.eval()
    return gen, disc

def main():
    st.title("Green-GAN Streamlit App")
    st.write("Generate or test samples with Green-GAN")

    # Load models
    gen, disc = load_models()


    if st.button("Generate Random Sample"):
        z = torch.randn(1, LATENT_DIM).to(DEVICE)
        with torch.no_grad():
            generated_sample = gen(z).cpu().numpy()
        st.write("Generated Sample:", generated_sample)


    st.subheader("Discriminator Test")
    sample_input = st.text_area("Enter features separated by commas (length must match N_FEATURES):")
    if st.button("Test Discriminator") and sample_input:
        try:
            features = [float(x) for x in sample_input.split(",")]
            if len(features) != N_FEATURES:
                st.error(f"Input must have {N_FEATURES} features.")
            else:
                x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    score = disc(x).item()
                st.write(f"Discriminator score: {score:.4f}")
        except ValueError:
            st.error("Invalid input. Make sure all values are numeric.")

if __name__ == "__main__":
    main()
