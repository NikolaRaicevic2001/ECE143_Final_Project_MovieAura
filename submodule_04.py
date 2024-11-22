import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_SEQ_LEN = 20  # Max number of past movies to consider
EMBEDDING_DIM = 128 # Dimension of the movie embedding
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerRecModel(nn.Module):
    def __init__(self, num_movies, input_dim, embedding_dim, num_heads, num_layers):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, embedding_dim)  # Project features to embedding_dim
        self.positional_embedding = nn.Embedding(MAX_SEQ_LEN, embedding_dim)
        self.transformer = nn.Transformer(d_model=embedding_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.output_layer = nn.Linear(embedding_dim, num_movies)
        self.max_seq_len = MAX_SEQ_LEN

    def forward(self, features, input_ids=None):
        # Project features into embedding space
        projected_features = self.input_projection(features)
        
        # Determine sequence length
        seq_len = self.max_seq_len if input_ids is None else min(input_ids.size(1), self.max_seq_len)
        
        # Expand projected features to match sequence length
        projected_features = projected_features.unsqueeze(1).expand(-1, seq_len, -1)

        # Add positional embeddings
        pos_embeds = self.positional_embedding(torch.arange(seq_len, device=features.device))
        combined_embeds = projected_features + pos_embeds

        # Transpose for transformer (seq_len, batch_size, embedding_dim)
        combined_embeds = combined_embeds.permute(1, 0, 2)

        # Apply transformer
        transformer_output = self.transformer(combined_embeds, combined_embeds)

        # Output probabilities for next movie
        output = self.output_layer(transformer_output[-1])  # Last position
        return output

def get_movie_scores(model, movie_sequence, movie_to_index, device):
    model.eval()
    with torch.no_grad():
        # Convert movie titles to indices
        input_indices = [movie_to_index[movie] for movie in movie_sequence]
        
        # Create input tensor
        input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)
        
        # Get model output
        output = model(input_tensor)
        
        # Apply softmax to get probabilities
        probabilities = F.softmax(output, dim=1)
        
        return probabilities.squeeze().cpu().numpy()

if __name__ == "__main__":
    # Load the saved model
    model = TransformerRecModel(num_movies=NUM_MOVIES, input_dim=INPUT_DIM, embedding_dim=EMBEDDING_DIM, num_heads=4, num_layers=2, max_seq_len=MAX_SEQ_LEN)
    model.load_state_dict(torch.load('transformer_rec_model.pth'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Example sequence of movie titles (representing a user's watch history)
    movie_sequences = ['Three Colors: Red', 'The Godfather', 'The Shawshank Redemption', 'The Dark Knight', 'Pulp Fiction', 
                       'The Godfather: Part II', 'The Lord of the Rings: The Return of the King', 'The Lord of the Rings: The Fellowship of the Ring', 
                       'The Lord of the Rings: The Two Towers', 'The Matrix']

    # Assuming you have a dictionary mapping movie titles to indices
    movie_to_index = {...}  # You need to provide this mapping

    print("Getting movie scores...")
    probabilities = get_movie_scores(model, movie_sequences, movie_to_index, device)
    print(probabilities)