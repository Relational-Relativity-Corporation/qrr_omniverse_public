"""
QRR Marine Simulation for Omniverse - Standalone Version
=========================================================
Everything in one file - no external imports needed
Demonstrates Quantum Relational Relativity mathematics for marine dynamics

Author: Robin B. Macomber (Relational Relativity LLC)
Date: November 2025
MODIFIED: Linear wave pattern instead of radial (boat-independent waves)
"""

import omni
from pxr import Usd, UsdGeom, UsdShade, UsdLux, Sdf, Gf, UsdPhysics, PhysxSchema
import numpy as np
import carb
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

# ============================================================================
# QRR MATHEMATICS - EMBEDDED
# ============================================================================

class QRRConstants:
    """Physical and mathematical constants"""
    HBAR = 1.054571817e-34
    COHERENCE_THRESHOLD = 0.95
    DECOHERENCE_RATE = 0.01
    DEFAULT_TIME_STEP = 0.01

class MathPrimitives:
    """Core mathematical operations"""
    
    @staticmethod
    def normalize_vector(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v if norm == 0 else v / norm
    
    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        return np.linalg.norm(np.array(a) - np.array(b))

class CoherenceField:
    """Coherence field with wave evolution"""
    
    def __init__(self, field_size: Tuple[int, int], initial_coherence: float = 1.0):
        self.field_size = field_size
        self.field_state = np.ones(field_size) * initial_coherence
        self.velocity_field = np.zeros(field_size)
        
    def compute_gradient(self) -> Tuple[np.ndarray, np.ndarray]:
        grad_y, grad_x = np.gradient(self.field_state)
        return grad_x, grad_y
    
    def compute_laplacian(self) -> np.ndarray:
        laplacian = np.zeros_like(self.field_state)
        laplacian[1:-1, 1:-1] = (
            self.field_state[:-2, 1:-1] +
            self.field_state[2:, 1:-1] +
            self.field_state[1:-1, :-2] +
            self.field_state[1:-1, 2:] -
            4 * self.field_state[1:-1, 1:-1]
        )
        return laplacian
    
    def evolve_wave(self, time_step: float, wave_speed: float = 1.0):
        laplacian = self.compute_laplacian()
        self.velocity_field += time_step * wave_speed**2 * laplacian
        self.field_state += time_step * self.velocity_field
        self.velocity_field *= 0.995
    
    def get_field_coherence(self) -> float:
        return float(np.mean(self.field_state))
    
    def get_field_variance(self) -> float:
        return float(np.var(self.field_state))

class RelationalEntity:
    """Entity in relational space"""
    
    def __init__(self, entity_id: str, weight: float = 1.0,
                 position: Optional[np.ndarray] = None):
        self.id = entity_id
        self.weight = weight
        self.position = position if position is not None else np.zeros(3)
        self.velocity = np.zeros(3)
        self.coherence_history = deque(maxlen=100)
        self.entanglement_partners: Dict[str, float] = {}
        
    def update_coherence(self, coherence_value: float):
        self.coherence_history.append(coherence_value)
        
    def add_entanglement(self, partner_id: str, strength: float):
        self.entanglement_partners[partner_id] = strength
        
    def get_average_coherence(self) -> float:
        return float(np.mean(self.coherence_history)) if self.coherence_history else 0.0

class CoherenceDensityCalculator:
    """Calculate coherence density"""
    
    @staticmethod
    def compute_coherence_density(system_energy: float,
                                 relational_bandwidth: float) -> float:
        if relational_bandwidth == 0:
            raise ValueError("Relational bandwidth cannot be zero")
        return system_energy / relational_bandwidth
    
    @staticmethod
    def coherence_decay(initial_coherence: float, decay_rate: float,
                       time: float) -> float:
        return initial_coherence * np.exp(-decay_rate * time)

class QRRSystem:
    """Integrated QRR system"""
    
    def __init__(self, system_name: str = "QRR_System"):
        self.system_name = system_name
        self.math = MathPrimitives()
        self.coherence_calc = CoherenceDensityCalculator()
        self.entities: Dict[str, RelationalEntity] = {}
        self.system_coherence = 1.0
        self.system_energy = 10.0
        self.relational_bandwidth = 5.0
        self.time = 0.0
        self.coherence_history = deque(maxlen=1000)
        self.energy_history = deque(maxlen=1000)
        
    def add_entity(self, entity: RelationalEntity) -> str:
        self.entities[entity.id] = entity
        return entity.id
    
    def create_entanglement(self, entity_a_id: str, entity_b_id: str,
                          strength: float = 1.0) -> bool:
        if entity_a_id not in self.entities or entity_b_id not in self.entities:
            return False
        self.entities[entity_a_id].add_entanglement(entity_b_id, strength)
        self.entities[entity_b_id].add_entanglement(entity_a_id, strength)
        return True
    
    def update_system_state(self, time_delta: float = None):
        if time_delta is None:
            time_delta = QRRConstants.DEFAULT_TIME_STEP
        
        self.time += time_delta
        
        coherence_density = self.coherence_calc.compute_coherence_density(
            self.system_energy, self.relational_bandwidth
        )
        
        self.system_coherence = self.coherence_calc.coherence_decay(
            coherence_density, QRRConstants.DECOHERENCE_RATE, time_delta
        )
        
        self.coherence_history.append(self.system_coherence)
        self.energy_history.append(self.system_energy)
        
        for entity in self.entities.values():
            entity.update_coherence(self.system_coherence)
    
    def get_system_metrics(self) -> Dict[str, float]:
        return {
            "system_coherence": float(self.system_coherence),
            "system_energy": float(self.system_energy),
            "relational_bandwidth": float(self.relational_bandwidth),
            "num_entities": len(self.entities),
            "average_coherence": float(np.mean(self.coherence_history)) if self.coherence_history else 0.0,
            "coherence_stability": 1.0 - float(np.std(self.coherence_history)) if len(self.coherence_history) > 1 else 1.0,
            "system_time": float(self.time)
        }

# ============================================================================
# MARINE SIMULATION
# ============================================================================

class QRRMarineSimulation:
    """Marine simulation using QRR mathematics"""
    
    def __init__(self, stage_path="/World"):
        self.stage = omni.usd.get_context().get_stage()
        self.stage_path = stage_path
        
        # QRR System
        self.qrr_system = QRRSystem("Marine_Simulation")
        
        # Coherence field for water (32x32 grid)
        self.water_field = CoherenceField(field_size=(32, 32), initial_coherence=1.0)
        
        # Simulation parameters
        self.grid_size = 32
        self.grid_spacing = 2.0
        self.wave_speed = 1.5
        self.wave_amplitude = 1.2
        self.time = 0.0
        self.dt = 0.016
        
        # USD references
        self.water_mesh = None
        self.boat_prim = None
        self.boat_entity = None
        self.boat_translate_op = None
        self.camera_prim = None

        # Animation control
        self.is_playing = False
        self.update_subscription = None
        self.frame_count = 0
        self.last_fps_time = 0
        self.fps = 0.0

        # AI sensor data
        self.sensor_data = {
            "wave_gradient": np.zeros(2),
            "boat_orientation": np.zeros(3),
            "velocity": np.zeros(3),
            "coherence_local": 1.0
        }
        
        print("[QRR Marine] Simulation initialized")
        
    def create_scene(self):
        """Create USD scene with water and boat"""
        print("[QRR Marine] Creating scene...")

        world = UsdGeom.Xform.Define(self.stage, self.stage_path)
        self._create_water_surface()
        self._create_boat()
        self._create_camera()
        self._create_lighting()
        self._setup_physics()

        print("[QRR Marine] Scene created successfully")
        
    def _create_water_surface(self):
        """Create water surface mesh"""
        water_path = f"{self.stage_path}/WaterSurface"
        self.water_mesh = UsdGeom.Mesh.Define(self.stage, water_path)
        
        # Generate grid
        vertices = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = (i - self.grid_size/2) * self.grid_spacing
                y = 0.0
                z = (j - self.grid_size/2) * self.grid_spacing
                vertices.append(Gf.Vec3f(x, y, z))
        
        # Generate faces
        indices = []
        for i in range(self.grid_size - 1):
            for j in range(self.grid_size - 1):
                idx = i * self.grid_size + j
                indices.extend([idx, idx + 1, idx + self.grid_size])
                indices.extend([idx + 1, idx + self.grid_size + 1, idx + self.grid_size])
        
        face_counts = [3] * (len(indices) // 3)
        
        self.water_mesh.GetPointsAttr().Set(vertices)
        self.water_mesh.GetFaceVertexIndicesAttr().Set(indices)
        self.water_mesh.GetFaceVertexCountsAttr().Set(face_counts)

        # Ensure visibility
        water_prim = self.water_mesh.GetPrim()
        water_prim.GetAttribute('visibility').Set('inherited')

        # Apply water material - blue, smooth, slightly reflective
        self._apply_material(water_path, color=(0.1, 0.4, 0.9), roughness=0.2, metallic=0.1)
        print(f"[QRR Marine] Water surface: {len(vertices)} vertices")
        
    def _create_boat(self):
        """Create box boat"""
        boat_path = f"{self.stage_path}/Boat"
        cube = UsdGeom.Cube.Define(self.stage, boat_path)
        cube.GetSizeAttr().Set(2.0)

        # Set transform - clear any existing ops first to avoid duplicates
        xform = UsdGeom.Xformable(cube)
        xform.ClearXformOpOrder()
        self.boat_translate_op = xform.AddTranslateOp()
        self.boat_translate_op.Set(Gf.Vec3d(0, 1.5, 0))

        self.boat_prim = cube.GetPrim()

        # Ensure visibility
        if not self.boat_prim.HasAttribute('visibility'):
            self.boat_prim.CreateAttribute('visibility', Sdf.ValueTypeNames.Token).Set('inherited')
        else:
            self.boat_prim.GetAttribute('visibility').Set('inherited')

        # Enable double-sided rendering
        gprim = UsdGeom.Gprim(self.boat_prim)
        gprim.GetDoubleSidedAttr().Set(True)

        # Apply boat material FIRST - orange, matte finish
        print(f"[QRR Marine] Applying orange material to boat at {boat_path}")
        self._apply_material(boat_path, color=(1.0, 0.5, 0.1), roughness=0.8, metallic=0.0)

        self.boat_entity = RelationalEntity("boat", weight=1000.0,
                                           position=np.array([0, 1.5, 0]))
        self.qrr_system.add_entity(self.boat_entity)

        print(f"[QRR Marine] Boat created at position (0, 1.5, 0)")

    def _create_camera(self):
        """Create and position camera to view the scene"""
        camera_path = f"{self.stage_path}/Camera"
        camera = UsdGeom.Camera.Define(self.stage, camera_path)

        # Set camera properties
        camera.GetFocalLengthAttr().Set(35.0)
        camera.GetFocusDistanceAttr().Set(100.0)

        # Position camera to view water and boat
        # Camera at (30, 20, 30) looking at origin
        # Clear any existing ops first to avoid duplicates
        xform = UsdGeom.Xformable(camera)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(30, 20, 30))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(-25, 45, 0))

        self.camera_prim = camera.GetPrim()

        # Set as active viewport camera
        viewport = omni.kit.viewport.utility.get_active_viewport()
        if viewport:
            viewport.camera_path = camera_path
            print(f"[QRR Marine] Camera created and set as active: {camera_path}")
        else:
            print(f"[QRR Marine] Camera created: {camera_path}")

    def _create_lighting(self):
        """Add lighting"""
        # Main directional light (sun)
        light_path = f"{self.stage_path}/DistantLight"
        light = UsdLux.DistantLight.Define(self.stage, light_path)
        light.CreateIntensityAttr(5000)  # Increased intensity
        light.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.95, 0.9))  # Warm sunlight

        xform = UsdGeom.Xformable(light)
        xform.ClearXformOpOrder()
        xform.AddRotateXYZOp().Set(Gf.Vec3f(-45, 45, 0))

        # Dome light for ambient lighting
        dome_light_path = f"{self.stage_path}/DomeLight"
        dome_light = UsdLux.DomeLight.Define(self.stage, dome_light_path)
        dome_light.CreateIntensityAttr(1000)
        dome_light.CreateColorAttr().Set(Gf.Vec3f(0.7, 0.8, 1.0))  # Sky blue ambient

        print("[QRR Marine] Lighting created (Distant + Dome)")
        
    def _setup_physics(self):
        """Setup PhysX"""
        scene_path = f"{self.stage_path}/PhysicsScene"
        scene = UsdPhysics.Scene.Define(self.stage, scene_path)
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, -1.0, 0.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        
        if self.boat_prim:
            rigid_body = UsdPhysics.RigidBodyAPI.Apply(self.boat_prim)
            mass_api = UsdPhysics.MassAPI.Apply(self.boat_prim)
            mass_api.CreateMassAttr().Set(1000.0)
            
        print("[QRR Marine] Physics setup complete")
        
    def _apply_material(self, prim_path, color=(0.5, 0.5, 0.5), roughness=0.4, metallic=0.0):
        """Apply UsdShade material with proper shader"""
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim:
            print(f"[QRR Marine] ✗ Warning: Prim not found at {prim_path}")
            return

        try:
            # Create material
            material_path = f"{prim_path}/Material"
            material = UsdShade.Material.Define(self.stage, material_path)

            # Create shader
            shader_path = f"{material_path}/Shader"
            shader = UsdShade.Shader.Define(self.stage, shader_path)
            shader.CreateIdAttr("UsdPreviewSurface")

            # Set shader parameters
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)

            # Connect shader output to material
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

            # Bind material to geometry
            binding_api = UsdShade.MaterialBindingAPI(prim)
            binding_api.Bind(material)

            # Also set displayColor as fallback
            gprim = UsdGeom.Gprim(prim)
            if gprim:
                gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])

            print(f"[QRR Marine] ✓ Material applied: {prim_path.split('/')[-1]} = RGB{color}, roughness={roughness}")
        except Exception as e:
            print(f"[QRR Marine] ✗ Error applying material to {prim_path}: {e}")

    def configure_viewport(self):
        """Configure viewport settings for optimal visibility"""
        try:
            # Set the camera as active
            camera_path = f"{self.stage_path}/Camera"
            viewport = omni.kit.viewport.utility.get_active_viewport()
            if viewport:
                viewport.camera_path = camera_path
                print(f"[QRR Marine] Viewport camera set to: {camera_path}")

            # Enable grid and axis display for reference
            settings = carb.settings.get_settings()
            settings.set("/app/viewport/grid/enabled", True)
            settings.set("/app/viewport/outline/enabled", True)

            print("[QRR Marine] Viewport configured")
            return True
        except Exception as e:
            print(f"[QRR Marine] Warning: Could not configure viewport: {e}")
            return False
            
    def update_wave_dynamics(self):
        """Update water using QRR coherence field - LINEAR WAVE PATTERN"""
        self.water_field.evolve_wave(self.dt, self.wave_speed)

        if self.water_mesh:
            vertices = []
            max_y = -999
            min_y = 999

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    x = (i - self.grid_size/2) * self.grid_spacing
                    
                    # LINEAR wave pattern - waves move along X axis (like ocean swells)
                    # Independent of boat position - consistent environmental waves
                    wave_phase = i * 0.5 - self.wave_speed * self.time
                    y = self.wave_amplitude * np.sin(wave_phase) * self.water_field.field_state[i, j]

                    max_y = max(max_y, y)
                    min_y = min(min_y, y)

                    z = (j - self.grid_size/2) * self.grid_spacing
                    vertices.append(Gf.Vec3f(x, y, z))

            self.water_mesh.GetPointsAttr().Set(vertices)

            # Debug output every 60 frames
            if self.frame_count % 60 == 0 and self.frame_count > 0:
                print(f"[Wave Update] Y range: [{min_y:.3f}, {max_y:.3f}] at t={self.time:.2f}s | LINEAR PATTERN")
            
    def update_boat_physics(self):
        """Update boat using QRR dynamics"""
        if not self.boat_entity:
            return
            
        boat_pos = self.boat_entity.position
        
        grid_x = int((boat_pos[0] / self.grid_spacing) + self.grid_size / 2)
        grid_z = int((boat_pos[2] / self.grid_spacing) + self.grid_size / 2)
        
        grid_x = np.clip(grid_x, 1, self.grid_size - 2)
        grid_z = np.clip(grid_z, 1, self.grid_size - 2)
        
        # Compute wave gradient
        gradient = self.water_field.compute_gradient()
        if len(gradient) == 2:
            wave_gradient_x = gradient[0][grid_x, grid_z]
            wave_gradient_z = gradient[1][grid_x, grid_z]
            self.sensor_data["wave_gradient"] = np.array([wave_gradient_x, wave_gradient_z])
        
        # Get water height using LINEAR wave pattern (matching update_wave_dynamics)
        wave_phase = grid_x * 0.5 - self.wave_speed * self.time
        water_height = self.wave_amplitude * np.sin(wave_phase) * self.water_field.field_state[grid_x, grid_z]
        
        # QRR buoyancy
        local_coherence = self.water_field.field_state[grid_x, grid_z]
        
        # Update boat height
        target_height = water_height + 1.0
        boat_pos[1] += (target_height - boat_pos[1]) * 0.1
        
        self.boat_entity.position = boat_pos
        self.boat_entity.update_coherence(local_coherence)
        
        # Update USD
        if self.boat_translate_op:
            self.boat_translate_op.Set(Gf.Vec3d(float(boat_pos[0]), float(boat_pos[1]), float(boat_pos[2])))
            
        self.sensor_data["boat_orientation"] = boat_pos
        self.sensor_data["coherence_local"] = local_coherence
        
    def ai_pilot_decision(self):
        """AI pilot navigation"""
        wave_gradient = self.sensor_data["wave_gradient"]
        local_coherence = self.sensor_data["coherence_local"]
        
        if self.time % 2.0 < self.dt:
            print(f"[AI Pilot] t={self.time:.1f}s | Gradient: [{wave_gradient[0]:.3f}, {wave_gradient[1]:.3f}] | Coherence: {local_coherence:.3f}")
            
    def step(self):
        """Single simulation step"""
        import time as time_module

        # FPS tracking
        current_time = time_module.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
            print(f"[QRR Marine] FPS: {self.fps:.1f} | Time: {self.time:.2f}s | Coherence: {self.qrr_system.system_coherence:.4f}")

        self.frame_count += 1

        self.qrr_system.update_system_state(self.dt)
        self.update_wave_dynamics()
        self.update_boat_physics()
        self.ai_pilot_decision()
        self.time += self.dt
        
    def _on_update(self, event):
        """Callback for Omniverse update events"""
        if self.is_playing:
            self.step()

    def play(self):
        """Start continuous animation"""
        if not self.is_playing:
            self.is_playing = True
            import time as time_module
            self.last_fps_time = time_module.time()
            self.frame_count = 0

            # Subscribe to update events
            try:
                update_stream = omni.kit.app.get_app().get_update_event_stream()
                self.update_subscription = update_stream.create_subscription_to_pop(
                    self._on_update, name="QRR Marine Update"
                )
                print("[QRR Marine] ▶ Animation STARTED - watching for updates...")
                print("[QRR Marine] If you don't see FPS updates in console, animation is not running")
            except Exception as e:
                print(f"[QRR Marine] ✗ ERROR starting animation: {e}")
                self.is_playing = False
        else:
            print("[QRR Marine] Already playing")

    def pause(self):
        """Pause animation"""
        if self.is_playing:
            self.is_playing = False
            if self.update_subscription:
                self.update_subscription.unsubscribe()
                self.update_subscription = None
            print(f"[QRR Marine] ⏸ Animation PAUSED at t={self.time:.2f}s")
        else:
            print("[QRR Marine] Already paused")

    def reset(self):
        """Reset simulation"""
        self.pause()
        self.time = 0.0
        self.water_field = CoherenceField(field_size=(32, 32), initial_coherence=1.0)
        if self.boat_entity:
            self.boat_entity.position = np.array([0, 1.5, 0])
        print("[QRR Marine] 🔄 Simulation RESET")

    def print_status(self):
        """Print simulation status"""
        metrics = self.qrr_system.get_system_metrics()
        print("\n" + "="*60)
        print("QRR MARINE SIMULATION STATUS")
        print("="*60)
        print(f"Status: {'▶ PLAYING' if self.is_playing else '⏸ PAUSED'}")
        print(f"FPS: {self.fps:.1f}")
        print(f"Time: {self.time:.2f}s")
        print(f"System Coherence: {metrics['system_coherence']:.4f}")
        print(f"Water Field Coherence: {self.water_field.get_field_coherence():.4f}")
        print(f"Boat Position: {self.boat_entity.position}")
        print(f"Local Coherence: {self.sensor_data['coherence_local']:.4f}")
        print("="*60 + "\n")

# ============================================================================
# EXECUTION
# ============================================================================

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("QRR MARINE SIMULATION - Omniverse + Quantum Relational Relativity")
    print("Relational Relativity LLC - Robin B. Macomber")
    print("WAVE MODE: Linear (ocean swells) - boat-independent waves")
    print("="*70 + "\n")

    sim = QRRMarineSimulation()
    sim.create_scene()
    sim.configure_viewport()

    print("\n" + "="*70)
    print("CONTROLS:")
    print("  sim.play()   - Start animation")
    print("  sim.pause()  - Pause animation")
    print("  sim.reset()  - Reset to beginning")
    print("  sim.print_status() - Show current status")
    print("="*70 + "\n")

    # Auto-start animation
    print("[QRR Marine] Starting animation in 2 seconds...")
    import time
    time.sleep(2)

    sim.play()

    return sim

if __name__ == "__main__":
    sim = main()