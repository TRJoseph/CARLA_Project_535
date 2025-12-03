from llama_cpp import Llama

#Yussef
model_location = r"C:\Users\Yussef\Documents\AIModel\openai_gpt-oss-20b-Q4_K_M.gguf" 
#Thomas
#model_location = r""

print(f"Loading model from {model_location}...")

llm = Llama(
    model_path=model_location,
    n_gpu_layers=-1, 
    n_ctx=131072,
    verbose=False
)

messages = [
    {"role": "system", "content": """You are an expert driver who needs to avoid obstacles while driving towards a
        desired path. A list of objects will be given to you with their positions and velocities, and size (x, y, v, heading, radius). The position and
        velocity of the ego vehicle will also be given, as well as a the current trajectory. You must explain if
        there are any critical objects that could cause collisions. If there are, you must modify the trajectory
        to ensure safety. while continuing along the road. The trajectory consists of 6 points over 3 seconds with
        equal spacing (0.5 seconds between). Your response should end with this final trajectory, and should be formatted as ((x, y, v, heading), ...) 
        Units are in meters and meter per second and radians. Heading of zero means facing along the positive x axis.
        If you must deviate for safety, ensure the trajectory is smooth and feasible for a car to follow. Also, try to return to the original trajectory"""},
    {"role": "user", "content": "Objects: ((5, -5, 10,pi/2, 3), (0, 5, 10, 0, 3)) Current Ego Position and Velocity: (0, 0, 10) Current Trajectory: ((5, 0, 10, 0), (10, 0, 10, 0), (15, 0, 10, 0), (20, 0, 10, 0), (25, 0, 10, 0), (30, 0, 10, 0))"},
]

output = llm.create_chat_completion(
    messages=messages,
    max_tokens=131072,
    temperature=0.7
)

print("\n--- Response ---")
print(output["choices"][0]["message"]["content"])