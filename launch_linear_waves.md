# Launch Instructions - QRR Marine Simulation (linear waves version)

**Save this file - Launch commands for linear wave demo**

Date: November 13, 2025
System: Windows Desktop with GTX 1050 Ti (4GB VRAM)

---

## Quick Launch (Copy/Paste into PowerShell)

```powershell
# Navigate to kit-app-template
cd D:\omniverse-marine-sim\projects\kit-app-template

# Launch Omniverse
.\repo.bat launch

# Wait for Omniverse to open, then in Script Editor, run:
exec(open('D:/omniverse-marine-sim/qrr-marine-omniverse/marine_sim_linear_waves.py').read())
```

---

## What's Different in This Version?

### Linear Wave Pattern (Ocean Swells)
- **OLD VERSION**: Circular waves radiating from center point
- **NEW VERSION**: Parallel wave bands moving across X-axis
- Waves are completely independent of boat position
- Boat responds to environmental waves (not generating them)

### Visual Effect
- You'll see parallel wave "ridges" moving across the water surface
- Boat bobs up and down as waves pass underneath
- More realistic ocean swell behavior
- Better for demonstrating AI pilot responding to environment

---

## Detailed Step-by-Step

### 1. Open PowerShell
- Press `Win + X`, select "Windows PowerShell" or "Terminal"

### 2. Navigate to Kit App Template
```powershell
cd D:\omniverse-marine-sim\projects\kit-app-template
```

### 3. Launch Omniverse Kit
```powershell
.\repo.bat launch
```

Wait for Omniverse window to open (takes ~30-60 seconds)

### 4. Open Script Editor in Omniverse
- Menu: `Window` → `Script Editor`
- Or look for Python console at bottom of window

### 5. Load and Run LINEAR WAVES Marine Simulation
In the Script Editor, paste and run:

```python
exec(open('D:/omniverse-marine-sim/qrr-marine-omniverse/marine_sim_linear_waves.py').read())
```

### 6. Verify Simulation Running
You should see:
- Blue water surface (32x32 grid)
- **Parallel wave bands moving across surface**
- Orange boat cube floating
- Console output showing "LINEAR PATTERN" in debug messages
- Boat bobbing as waves pass underneath

---

## File Locations

### Linear Waves Simulation Script
```
D:\omniverse-marine-sim\qrr-marine-omniverse\marine_sim_linear_waves.py
```

### Original Radial Waves Version (Backup)
```
D:\omniverse-marine-sim\qrr-marine-omniverse\omniverse_marine_standalone.py
```

### Omniverse Kit Installation
```
D:\omniverse-marine-sim\projects\kit-app-template\
```

---

## Comparison: Radial vs Linear Waves

### Radial Waves (Original)
```python
# Circular propagation from center
r = np.sqrt(dx**2 + dy**2)
wave_phase = r - self.wave_speed * self.time
```

### Linear Waves (New)
```python
# Parallel bands along X-axis
wave_phase = i * 0.5 - self.wave_speed * self.time
```

---

## Console Output Differences

### Linear Waves Version Shows:
```
[QRR Marine] WAVE MODE: Linear (ocean swells) - boat-independent waves
[Wave Update] Y range: [-0.300, 0.300] at t=2.00s | LINEAR PATTERN
```

### Original Version Shows:
```
[Wave Update] Y range: [-0.300, 0.300] at t=2.00s
```

---

## Troubleshooting

### "Cannot find marine_sim_linear_waves.py"
Verify the file exists:
```powershell
dir "D:\omniverse-marine-sim\qrr-marine-omniverse\marine_sim_linear_waves.py"
```

### Want to Switch Back to Original?
Just change the script path in Script Editor:
```python
exec(open('D:/omniverse-marine-sim/qrr-marine-omniverse/omniverse_marine_standalone.py').read())
```

### Waves Look the Same?
- Watch carefully - linear waves move as bands across the surface
- Radial waves expand outward from a center point
- Console output should say "LINEAR PATTERN"

---

## Demo Talking Points

### Why Linear Waves Matter for AI
1. **Predictable Environment**: AI can learn wave patterns more easily
2. **Realistic Ocean Conditions**: Ocean swells move in dominant directions
3. **Navigation Training**: AI learns to pilot through consistent wave fields
4. **Collision Avoidance**: Clear directional wave motion for obstacle detection

### QRR Advantages (Same in Both Versions)
- O(N) complexity vs O(N²) traditional CFD
- Coherence field evolution for wave dynamics
- Relationships as primitives, not particles
- 10-20x performance improvement

---

## Performance Notes

**Both versions run identically:**
- GPU: NVIDIA GeForce GTX 1050 Ti (4GB VRAM)
- Grid: 32x32 water surface
- FPS: ~14-15 FPS (acceptable for demo)
- Memory: ~1.3 GB process memory, ~441 MB GPU memory

---

## Quick Switch Guide

### To Launch Linear Waves:
```python
exec(open('D:/omniverse-marine-sim/qrr-marine-omniverse/marine_sim_linear_waves.py').read())
```

### To Launch Original (Radial):
```python
exec(open('D:/omniverse-marine-sim/qrr-marine-omniverse/omniverse_marine_standalone.py').read())
```

### Both Are Safe
- Non-destructive - can switch any time
- Same QRR mathematics core
- Same performance characteristics
- Just different wave patterns

---

## Controls (Same for Both Versions)

```
sim.play()          - Start animation
sim.pause()         - Pause animation
sim.reset()         - Reset to beginning
sim.print_status()  - Show current status
```

---

**Save Location:**
```
D:\omniverse-marine-sim\qrr-marine-omniverse\launch_linear_waves.md
```

---

*Relational Relativity LLC - Robin B. Macomber & Bruce Stevenson*  
*Linear Waves Version: November 13, 2025*  
*Demo Client: Pollentia Inc.*
