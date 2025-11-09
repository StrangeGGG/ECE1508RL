import highway_env
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

print("Verify environment installation...")

try:
    # Test basic environment
    env = gym.make('intersection-v0')
    print("highway_env Installation successful!")

    # Testing NumPy and Matplotlib
    arr = np.array([1, 2, 3])
    print(f"numpy working fine: {arr * 2}")

    # Simple drawing test
    plt.figure(figsize=(6, 4))
    plt.plot([1, 2, 3], [1, 4, 9])
    plt.title("Environmental testing")
    plt.savefig('./environment_test.png')
    plt.close()
    print("matplotlib working fine!")

    env.close()

except Exception as e:
    print(f"Environment test failed: {e}")