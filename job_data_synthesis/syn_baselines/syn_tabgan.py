import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split

df = pd.read_csv('/syn_eval_mle/syn_input_job.csv')

# Define the Generator network
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()  # Output values between -1 and 1, will be scaled later
        )
    
    def forward(self, z):
        return self.model(z)

# Define the Discriminator network with gradient penalty
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1)
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    interpolates = interpolates.requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones(real_samples.size(0), 1)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_tabgan(generator, discriminator, train_data, val_data, num_epochs=100, batch_size=500, lr=0.0002, lambda_gp=10):
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    val_data_scaled = scaler.transform(val_data)
    
    for epoch in range(num_epochs):
        for i in range(0, len(train_data_scaled), batch_size):
            # Train discriminator with real data
            discriminator.zero_grad()
            idx = np.random.randint(0, len(train_data_scaled), batch_size)
            real_batch = torch.tensor(train_data_scaled[idx], dtype=torch.float32)
            real_labels = torch.ones(batch_size, 1)
            output_real = discriminator(real_batch)
            loss_real = criterion(output_real, real_labels)
            
            # Train discriminator with generated data
            z = torch.randn(batch_size, latent_dim)
            fake_data = generator(z)
            fake_labels = torch.zeros(batch_size, 1)
            output_fake = discriminator(fake_data.detach())
            loss_fake = criterion(output_fake, fake_labels)
            
            # Compute gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_batch, fake_data)
            d_loss = loss_real + loss_fake + lambda_gp * gradient_penalty
            d_loss.backward()
            optimizer_d.step()
            
            # Train generator
            generator.zero_grad()
            z = torch.randn(batch_size, latent_dim)
            gen_data = generator(z)
            output_gen = discriminator(gen_data)
            g_loss = criterion(output_gen, real_labels)
            
            # Diversity regularization
            diversity_loss = torch.mean(torch.std(gen_data, dim=0))
            g_loss -= 0.1 * diversity_loss  # Add diversity regularization term

            g_loss.backward()
            optimizer_g.step()
            
        # Validation performance check
        with torch.no_grad():
            val_z = torch.randn(len(val_data_scaled), latent_dim)
            val_fake_data = generator(val_z)
            val_output_fake = discriminator(val_fake_data)
            val_fake_labels = torch.zeros(len(val_data_scaled), 1)
            val_d_loss = criterion(val_output_fake, val_fake_labels).item()

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}, val_d_loss: {val_d_loss}, diversity_loss: {diversity_loss.item()}")
    
    return generator, discriminator

train_data, temp_data = train_test_split(df, test_size=0.3, random_state=0)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=0)

scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
val_data_scaled = scaler.transform(val_data)
test_data_scaled = scaler.transform(test_data)

# Define dimensions
input_dim = df.shape[1]  # Number of features
output_dim = input_dim   # Output dimension matches input dimension

# Define hyperparameters
latent_dim = 32
num_epochs = 50
batch_size = 100
lr = 0.0002

generator = Generator(latent_dim, output_dim)
discriminator = Discriminator(input_dim)

generator, discriminator = train_tabgan(generator, discriminator, train_data, val_data, num_epochs=num_epochs, batch_size=batch_size, lr=lr)

# Generate synthetic data
num_generated_rows = 1000
z = torch.randn(num_generated_rows, latent_dim)
synthetic_data = generator(z).detach().numpy()
synthetic_data = scaler.inverse_transform(synthetic_data)

synthetic_df = pd.DataFrame(data=synthetic_data, columns=df.columns)
synthetic_df = synthetic_df.round().astype(int)
synthetic_df = synthetic_df.applymap(lambda x: max(x, -x))

synthetic_df.to_csv('out_tabgan.csv', index=False)
print(synthetic_df.head())
