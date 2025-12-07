from llama_cpp import Llama

# 1. SETUP
model_location = r"C:\Users\Yussef\Documents\AIModel\gemma-2-9b-it-Q4_K_M.gguf" 

print(f"Loading model from {model_location}...")

llm = Llama(
    model_path=model_location,
    n_gpu_layers=-1, 
    n_ctx=8192,
    verbose=False
)

# 2. PROMPT ENGINEERING (Adjusted for WORLD FRAME)
system_instruction = """You are a trajectory optimization solver for an autonomous vehicle.
TASK: Check for collisions and modify the trajectory if necessary.

PHYSICS CONTEXT:
- Coordinates are in the GLOBAL WORLD FRAME (Absolute x, y, heading).
- The trajectory contains N=6 points.
- Time step (dt) is 0.5 seconds.
- Total prediction horizon is 3.0 seconds.
- Units are m, m/s, degrees

INPUT DATA:
- Obstacles: List of (x, y, v, heading, radius).
- Ego State: (x, y, v, heading).
- Trajectory: List of 6 tuples ((x, y, v, heading), ...).

INSTRUCTIONS:
1. RELATIVE DISTANCE: Calculate distance = (Object_Pos - Ego_Pos).
2. PREDICT: Estimate where obstacles will be in World Coordinates over 3s.
3. CHECK: If an obstacle intersects the path (margin = Radius + 2m), modify the trajectory to move laterally to avoid it.
4. SMOOTH: Ensure the modified path is physically feasible. This includes a reasonable heading.
5. You must try to return to the original trajectory after avoiding obstacles.
6. OUTPUT: Return the final list of 6 tuple trajectory points as the last line in the response. Analysis text can precede it.
7. DOUBLE CHECK: Ensure no collisions remain in the final trajectory. If they do, adjust further.
"""

# 3. TEST DATA (World Frame: Obstacle is at x=520, Ego is at x=500 -> 20m distance)
user_input = (
    "Objects: ((515.0, 30.0, 0.0, 0.0, 2.0),) "
    "Ego: (500.0, 30.0, 10.0, 10.0) "
    "Trajectory: ((505.0, 30.0, 10.0, 0.0), (510.0, 30.0, 10.0, 0.0), (515.0, 30.0, 10.0, 0.0), "
    "(520.0, 30.0, 10.0, 0.0), (525.0, 30.0, 10.0, 0.0), (530.0, 30.0, 10.0, 0.0))"
)

# 4. MERGE
combined_prompt = f"{system_instruction}\n\nUSER DATA:\n{user_input}\n\nOutput:"

messages = [
    {"role": "user", "content": combined_prompt}
]

# 5. RUN
output = llm.create_chat_completion(
    messages=messages,
    max_tokens=8192,
    temperature=0.1, 
    top_p=0.7
)

print("\n--- Response ---")
print(output["choices"][0]["message"]["content"])