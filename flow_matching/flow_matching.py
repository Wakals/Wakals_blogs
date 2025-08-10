# flow_matching.py by Wakals
import math, torch, random, numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from tqdm import trange

# config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PARTICLES = 400
LINE_PARTICLES = 120
STEPS = 100
DT = 1.0 / STEPS
TRAIN_STEPS = 20000
HIDDEN = 128

def sample_spiral(n, noise=0.05, turns=2.5, a=0.35):
    θ   = torch.rand(n) * (2 * math.pi * turns)
    r   = a * θ
    pts = torch.stack([r * torch.cos(θ), r * torch.sin(θ)], dim=1)
    return pts + noise * torch.randn_like(pts)

def sample_pair(n):
    z = torch.randn(n, 2)
    y = sample_spiral(n)
    t = torch.rand(n, 1)
    x_t = (1 - t) * z + t * y
    v   = y - z
    return x_t, t, v

class VectorField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, HIDDEN), torch.nn.Tanh(),
            torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.Tanh(),
            torch.nn.Linear(HIDDEN, 2)
        )
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))  # here is how we feed time t

model = VectorField().to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"Training on {DEVICE} …")
for step in trange(TRAIN_STEPS):
    x_t, t, v_star = sample_pair(512)
    loss = ((model(x_t.to(DEVICE), t.to(DEVICE)) - v_star.to(DEVICE))**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

torch.save(model.state_dict(), "flow_matching_ckpt.pth")
print("\033[92mSaved flow_matching_ckpt.pth\033[0m")

@torch.no_grad()
def simulate(model, n=PARTICLES, steps=STEPS):
    z   = torch.randn(n, 2, device=DEVICE)
    x   = z.clone()
    traj = [x.cpu()]
    for s in range(steps):
        t = torch.full((n,1), (s+0.5)/steps, device=DEVICE)
        x  = x + model(x, t) * DT
        traj.append(x.cpu())
    return [p.numpy() for p in traj]

traj = simulate(model)  # list[frames][N,2]

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-6, 6); ax.set_ylim(-6, 6); ax.set_aspect('equal')
ax.set_title("Flow Matching :  Gaussian → Spiral")
ax.set_facecolor('#f8f8f8')

target_pts = sample_spiral(PARTICLES, noise=0.0).numpy()
ax.scatter(target_pts[:,0], target_pts[:,1],
           s=14, c='tab:red', alpha=0.35, label='target spiral')

scat = ax.scatter([], [], s=18, c='tab:blue', alpha=.9, label='particles')

lines = []
for _ in range(LINE_PARTICLES):
    ln, = ax.plot([], [], lw=0.7, color='gray', alpha=0.3)
    lines.append(ln)

ax.legend(loc='upper left', frameon=False)

def init():
    scat.set_offsets(traj[0])
    for ln in lines:
        ln.set_data([], [])
    ax.set_xlabel("t = 0.00")
    return [scat] + lines

def update(frame):
    scat.set_offsets(traj[frame])

    for idx, ln in enumerate(lines):
        path = np.array([step[idx] for step in traj[:frame+1]])
        ln.set_data(path[:,0], path[:,1])

    ax.set_xlabel(f"t = {frame / (len(traj)-1):.2f}")
    return [scat] + lines

ani = FuncAnimation(fig, update, frames=range(len(traj)),
                    init_func=init, interval=60, blit=False)


ani.save("flow_matching_animation.mp4",
            writer=FFMpegWriter(fps=15, bitrate=2000))
print("\033[92mSaved flow_matching_animation.mp4\033[0m")

ani.save("flow_matching_animation.gif",
            writer=PillowWriter(fps=15))
print("\033[92mSaved flow_matching_animation.gif\033[0m")

plt.show()
