# Quick Start Guide - QRR Marine Simulation

**Get the demo running in 5 minutes**

---

## Prerequisites

Before starting, ensure you have:

- ✅ NVIDIA Omniverse installed ([Download here](https://www.nvidia.com/en-us/omniverse/))
- ✅ NVIDIA GPU (GTX 1050 Ti or better recommended)
- ✅ This repository cloned to your machine

---

## Launch Steps

### Step 1: Start Omniverse Kit

Navigate to your Omniverse Kit installation directory and launch:

**Windows:**
```powershell
cd <your-omniverse-kit-path>
.\kit.exe
```

**Linux:**
```bash
cd <your-omniverse-kit-path>
./kit.sh
```

Wait for the Omniverse window to fully load (~30-60 seconds).

---

### Step 2: Open Script Editor

In the Omniverse window:
1. Click `Window` in the menu bar
2. Select `Script Editor`
3. A Python console should appear (usually at the bottom)

---

### Step 3: Load the Simulation

In the Script Editor console, paste and run:

```python
exec(open('<path-to-repo>/marine_sim_linear_waves.py').read())
```

Replace `<path-to-repo>` with your actual path, for example:
- **Windows**: `'C:/Projects/qrr_omniverse_public/marine_sim_linear_waves.py'`
- **Linux**: `'/home/user/qrr_omniverse_public/marine_sim_linear_waves.py'`

**Note:** Use forward slashes (`/`) even on Windows.

---

### Step 4: Verify It's Running

The simulation auto-starts after 2 seconds. You should see:

**Visual:**
- Blue water surface with animated waves
- Orange cube (boat) bobbing on the water
- Waves moving in parallel bands across the surface

**Console Output:**
```
[QRR Marine] FPS: 14.5 | Time: 2.00s | Coherence: 1.9832
[Wave Update] Y range: [-1.200, 1.200] at t=2.00s | LINEAR PATTERN
[AI Pilot] t=2.0s | Gradient: [0.023, -0.015] | Coherence: 0.998
```

---

## Controls

Once the simulation is running, you can control it from the Script Editor:

### Basic Controls

```python
# Pause the animation
sim.pause()

# Resume playing
sim.play()

# Reset to start
sim.reset()

# Show current status
sim.print_status()
```

### Manual Stepping (When Paused)

```python
# Advance one frame at a time
sim.step()

# Or advance multiple frames
for i in range(100):
    sim.step()
```

---

## Customizing Parameters

Want to see bigger waves? Faster motion? Edit the parameters!

### Step 1: Open the Python File

Open `marine_sim_linear_waves.py` in your text editor.

### Step 2: Find the Simulation Parameters

Look for the `__init__` method around line 225:

```python
# Simulation parameters
self.grid_size = 32
self.grid_spacing = 2.0
self.wave_speed = 1.5
self.wave_amplitude = 1.2    # ← Change this!
self.time = 0.0
self.dt = 0.016
```

### Step 3: Modify and Save

Try changing `wave_amplitude`:
- `0.3` = calm seas
- `1.2` = moderate waves (default)
- `2.0` = rough seas

Save the file.

### Step 4: Reload

In Omniverse Script Editor, just re-run the exec() command:

```python
exec(open('<path-to-repo>/marine_sim_linear_waves.py').read())
```

The new simulation will start with your updated parameters!

---

## Troubleshooting

### "Cannot find file"

**Problem:** Python can't locate your script file.

**Solution:** 
- Use the **absolute path** to the file
- Use **forward slashes** (`/`) even on Windows
- Check the path is correct: `import os; os.path.exists('<your-path>')`

---

### Black screen / Nothing visible

**Problem:** Camera might be in wrong position or scene not created.

**Solutions:**
- Try zooming out: Scroll mouse wheel backward
- Press `F` key to reset camera focus
- Check console for error messages
- Re-run the exec() command

---

### Boat not moving

**Problem:** Animation might not have started.

**Solutions:**
- Check console output - should show FPS updates every second
- Try manually: `sim.play()`
- If still stuck, check for error messages in console

---

### Low FPS / Poor performance

**Problem:** Hardware limitations or large grid size.

**Solutions:**
- Reduce `grid_size` from 32 to 16 (in the code)
- Close other GPU-intensive applications
- Ensure you're using an NVIDIA GPU, not integrated graphics
- Lower `wave_amplitude` slightly (less computation for large waves)

---

### Import errors

**Problem:** Missing Python packages.

**Solution:** 
This demo is designed to run in Omniverse's Python environment which includes all necessary packages (NumPy, etc.). Make sure you're running from within Omniverse Kit, not from a separate Python installation.

---

## What You Should See

### Wave Patterns

The simulation shows **linear waves** moving across the surface - like ocean swells. The waves move in parallel bands along the X-axis, creating a realistic ocean-like motion.

### Boat Motion

The orange cube represents a boat. It:
- Bobs up and down as waves pass underneath
- Follows the water surface height
- Generates sensor data (visible in console output)

### Console Output

Every second you should see:
```
[QRR Marine] FPS: 14.5 | Time: 5.00s | Coherence: 1.9832
```

Every 4 seconds you should see AI pilot data:
```
[AI Pilot] t=4.0s | Gradient: [0.023, -0.015] | Coherence: 0.998
```

---

## Next Steps

Once you have the basic demo running:

1. **Experiment with parameters** - Try different wave heights, speeds, grid sizes
2. **Study the code** - See how QRR mathematics are implemented
3. **Read the full README** - Understand the mathematical foundation
4. **Modify the simulation** - Add your own features or sensors

---

## Performance Expectations

### Tested Hardware

| Component | Spec | FPS |
|-----------|------|-----|
| GPU | GTX 1050 Ti (4GB) | ~15 fps |
| GPU | RTX 3060 (12GB) | ~60 fps |
| GPU | RTX 4090 (24GB) | ~120 fps |

Your performance will vary based on GPU capability.

---

## Getting Help

Having issues not covered here?

1. **Check the console** for error messages
2. **Review the full README.md** for detailed documentation
3. **Open an issue** on GitHub with:
   - Your GPU model
   - Omniverse version
   - Error messages from console
   - What you've already tried

---

## Summary

**TL;DR:**
1. Install Omniverse
2. Launch Omniverse Kit
3. Open Script Editor (`Window` → `Script Editor`)
4. Run: `exec(open('<path>/marine_sim_linear_waves.py').read())`
5. Watch waves and boat!

That's it! You should now have a working QRR marine simulation.

---

*For detailed technical information, see README.md*  
*Relational Relativity LLC - 2025*