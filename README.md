Here is the translation of your technical specifications into English:LWR Traffic Flow Model (1D Hyperbolic)
PDE: ∂tρ+∂x(ρ V(ρ))=0,V(ρ)=Vmax(1−ρ/ρmax).
Domain: x∈[0,1], t∈[0,1]; Vmax=1, ρmax=1.
Boundary Conditions (BC): Left boundary — inflow profile (ramp: ρ(t)=0.2→0.8); Right boundary — free outflow (non-characteristic condition).
Initial Conditions (IC): Step density profile (Riemann problem) and a smooth profile.
Numerical Reference: Godunov scheme (monotonic).
Test Cases: 3 inflow profiles (step / triangular / sinusoidal).
