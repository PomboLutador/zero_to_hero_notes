from __future__ import annotations

import torch
import torch.nn.functional as F


class NGramModel:
    def __init__(
        self,
        n_gram_size: int,
        vocab_size: int,
        embedding_size: int,
        hidden_layer_size: int,
        manual_seed: int = 2147483647,
    ) -> None:
        self.rng = torch.Generator().manual_seed(manual_seed)
        self.context_embedder = self.get_context_embedder(
            vocab_size=vocab_size, embedding_size=embedding_size
        )
        self.hidden_layer_weights, self.hidden_layer_biases = self.get_hidden_layer(
            embedding_size=embedding_size,
            n_gram_size=n_gram_size,
            hidden_layer_size=hidden_layer_size,
        )
        self.output_layer_weights, self.output_layer_biases = self.get_output_layer(
            hidden_layer_size=hidden_layer_size, vocab_size=vocab_size
        )
        self.params = [
            self.context_embedder,
            self.hidden_layer_weights,
            self.hidden_layer_biases,
            self.output_layer_weights,
            self.output_layer_biases,
        ]
        print("Total number of parameters:", sum(p.nelement() for p in self.params))
        self.set_requires_grad()
        self.losses = []
        self.iterations = []
        self.validation_losses = []
        self.validation_iterations = []

    def train(
        self,
        X_training: torch.Tensor,
        Y_training: torch.Tensor,
        X_validation: torch.Tensor,
        Y_validation: torch.Tensor,
        num_iterations: int,
        learning_rate: float,
        batch_size: int,
    ) -> None:
        last_iteration = max(self.iterations) if self.iterations else 0
        for i in range(last_iteration, last_iteration + num_iterations):
            ix = torch.randint(
                0, X_training.shape[0], (batch_size,), generator=self.rng
            )
            X_batch, Y_batch = X_training[ix], Y_training[ix]

            logits = self.get_logits(X_input=X_batch)
            loss = self.get_cross_entropy_loss(logits=logits, expected=Y_batch)
            self.backward(learning_rate=learning_rate, loss=loss)

            if i % (num_iterations / 10) == 0:
                print(f"Iteration {i} loss: {loss.data.item():.4f}")

            if i % (num_iterations / 100) == 0:
                self.validation_iterations.append(i)
                self.validation_losses.append(
                    self.evaluate_validation_loss(
                        X_validation=X_validation, Y_validation=Y_validation
                    )
                )

            self.iterations.append(i)
            self.losses.append(loss.data.item())
        print(f"Loss on the final iteration: {loss.data.item():.4f}")
        print(
            f"Validation loss on the final iteration: "
            f"{self.evaluate_validation_loss(X_validation=X_validation, Y_validation=Y_validation):.4f}"
        )

    @torch.no_grad()
    def evaluate_validation_loss(
        self, X_validation: torch.Tensor, Y_validation: torch.Tensor
    ) -> float:
        logits = self.get_logits(X_input=X_validation)
        return self.get_cross_entropy_loss(
            logits=logits, expected=Y_validation
        ).data.item()

    def get_logits(self, X_input: torch.Tensor) -> torch.Tensor:
        embeddings = self.context_embedder[X_input]
        embeddings_concatenated = embeddings.view(
            embeddings.shape[0], -1
        )  # keep number of rows, but concatenate across all other dimensions
        hidden_layer = torch.tanh(
            embeddings_concatenated @ self.hidden_layer_weights
            + self.hidden_layer_biases
        )
        self.hidden_layer = hidden_layer
        return hidden_layer @ self.output_layer_weights + self.output_layer_biases

    def get_cross_entropy_loss(
        self, logits: torch.Tensor, expected: torch.Tensor
    ) -> torch.Tensor:
        return F.cross_entropy(logits, expected)

    def backward(self, learning_rate: float, loss: torch.Tensor) -> None:
        for p in self.params:
            p.grad = None
        loss.backward()

        for p in self.params:
            p.data += -learning_rate * p.grad

    def get_context_embedder(self, vocab_size: int, embedding_size: int):
        """Get lookup table mapping n-grams to their embeddings.

        C.shape = vocab_size x embedding_size

        If x is training example of length 3 (i.e. a 3-gram, tensor([[0, 0, 20]]), for example)
        then C[x] will have dimension of 1 x 3 x embedding_size by picking out rows 0, 0, and 20
        from C.
        """
        return torch.randn((vocab_size, embedding_size), generator=self.rng)

    def get_hidden_layer(
        self, embedding_size: int, n_gram_size: int, hidden_layer_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = (
            torch.randn(
                (embedding_size * n_gram_size, hidden_layer_size), generator=self.rng
            )
            * 0.01
        )
        # initial weights should be relatively small such that after
        # taking the tanh, the output is not already saturated (i.e. close)
        # to the tails of the tanh where gradients can become almost zero
        # and neurons are completely 'dead'
        biases = torch.randn(hidden_layer_size, generator=self.rng) * 0.01
        return (weights, biases)

    def get_output_layer(
        self, hidden_layer_size: int, vocab_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = (
            torch.randn((hidden_layer_size, vocab_size), generator=self.rng) * 0.01
        )
        # initial logits should be small, such that after taking the exp
        # and normalizing, probabilities are almost equal for all possible
        # elements of the vocabulary
        biases = torch.randn(vocab_size, generator=self.rng) * 0.01
        return weights, biases

    def set_requires_grad(self):
        for p in self.params:
            p.requires_grad = True


if __name__ == "__main__":
    model = NGramModel(
        n_gram_size=3, vocab_size=27, embedding_size=100, hidden_layer_size=1000
    )
