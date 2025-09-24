# ML-Agents Data Collection Guide

## Core Data Types Available

### 1. Movement & Physics Data
- **Position/Velocity**: World/local coordinates, linear/angular velocity
- **Orientation**: Transform rotations, relative alignments to targets
- **Physics Forces**: Joint torques, collision forces, ground contact
- **Implementation**: `sensor.AddObservation()` or PhysicsBodySensor component

### 2. Spatial Awareness  
- **Distance Measurements**: Ray perception, proximity to objects/goals
- **Visual Data**: Camera observations, custom render textures
- **Grid-Based**: Occupancy grids, spatial relationships
- **Implementation**: RayPerceptionSensor, CameraSensor, GridSensor components

### 3. Behavioral Metrics
- **Task Performance**: Goal achievement, completion rates, efficiency
- **Learning Progress**: Custom analytics via StatsRecorder
- **Multi-Agent**: Inter-agent distances, cooperation metrics
- **Implementation**: Custom counters, episode-level analytics

## Collection Methods

### Vector Observations (Manual)
Collect specific data points in agent's `CollectObservations()` method:
```csharp
sensor.AddObservation(transform.position); // Position
sensor.AddObservation(rigidbody.velocity); // Velocity  
sensor.AddObservation(Vector3.Distance(pos, target)); // Distance
sensor.AddObservation(contactDetected);    // Boolean state
```

### Sensor Components (Automatic)
Attach components to GameObjects for automatic data collection:
- **RayPerceptionSensor**: Spatial awareness via raycasting
- **CameraSensor**: Visual observations from cameras
- **PhysicsBodySensor**: Comprehensive physics data from rigidbodies
- **BufferSensor**: Variable-length sequences

### Analytics Integration
Track custom metrics for research analysis:
```csharp
StatsRecorder.Add("Goal/Success", 1, StatAggregationMethod.Sum);
StatsRecorder.Add("Movement/Speed", velocity.magnitude);
```

## Key Implementation Examples

### Simple Physics Tracking (3DBall)
```csharp
// Project/Assets/ML-Agents/Examples/3DBall/Scripts/Ball3DAgent.cs:24-32
public override void CollectObservations(VectorSensor sensor)
{
    // Rotations
    sensor.AddObservation(transform.rotation.z);
    sensor.AddObservation(transform.rotation.x);     
    sensor.AddObservation(ball.transform.position - transform.position); // Relative position
    sensor.AddObservation(m_BallRb.velocity); // Ball physics
}
```

### Complex Locomotion (Crawler)  
```csharp
// Project/Assets/ML-Agents/Examples/Crawler/Scripts/CrawlerAgent.cs:136-165
// Note: Uses velocity matching and per-body-part observations
sensor.AddObservation(Vector3.Distance(velGoal, avgVel)); // Goal-directed movement
sensor.AddObservation(target.position);                   // Target awareness
foreach(var bodyPart in bodyParts) {
    sensor.AddObservation(bodyPart.groundContact.touchingGround); // Contact detection
    sensor.AddObservation(bodyPart.currentStrength / maxForce);   // Normalized forces
}
```

### Custom Analytics (Hallway)
```csharp
// Project/Assets/ML-Agents/Examples/Hallway/Scripts/HallwayAgent.cs:88-94
m_statsRecorder = Academy.Instance.StatsRecorder;
m_statsRecorder.Add("Goal/Correct", 1, StatAggregationMethod.Sum); // Success tracking
m_statsRecorder.Add("Goal/Wrong", 1, StatAggregationMethod.Sum);   // Failure analysis
```

