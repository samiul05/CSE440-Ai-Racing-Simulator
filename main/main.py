from game import Game
from config import TRAINING_EPISODES, GPU_AVAILABLE
import sys
import os
import pygame

def main():
    print("Self-Driving Car Racing Game with Reinforcement Learning")
    print("=" * 60)
    
    if GPU_AVAILABLE:
        print("✅ NVIDIA GPU DETECTED - Training will be accelerated!")
    else:
        print("⚠️  No NVIDIA GPU detected - Training will use CPU")
    
    print("\nOptions:")
    print("1. Train NEW model (Fast Training - No Visualization)")
    print("2. Train NEW model (With Visualization - Slower)")
    print("3. CONTINUE training existing model (Fast Training)")
    print("4. CONTINUE training existing model (With Visualization)")
    print("5. Test the trained agent")
    print("6. Train new model and then test")
    
    choice = input("\nEnter your choice (1/2/3/4/5/6): ").strip()
    
    game = None
    
    try:
        if choice == "1":
            # Train NEW model - Fast training without visualization
            print("\n🚀 Starting NEW MODEL training (No Visualization)...")
            game = Game(fast_training=True)
            episodes = int(input(f"Enter number of training episodes (default {TRAINING_EPISODES}): ") or TRAINING_EPISODES)
            game.train(episodes)
            game.agent.save_model("car_model.h5")
            print("✅ Model saved as car_model.h5")
            
        elif choice == "2":
            # Train NEW model - Training with visualization (slower)
            print("\n🎮 Starting NEW MODEL training with Visualization...")
            print("Press ESC or close window to stop training early")
            game = Game(fast_training=False)
            episodes = int(input(f"Enter number of training episodes (default {TRAINING_EPISODES}): ") or TRAINING_EPISODES)
            game.train(episodes)
            game.agent.save_model("car_model.h5")
            print("✅ Model saved as car_model.h5")
            
        elif choice == "3":
            # CONTINUE training existing model - Fast training
            print("\n🚀 CONTINUING training of existing model (No Visualization)...")
            game = Game(fast_training=True)
            
            # Try to load existing model
            if game.agent.load_model("car_model.h5"):
                print("✅ Continuing from existing model!")
                # Optionally adjust epsilon for continued exploration
                current_epsilon = game.agent.get_epsilon()
                print(f"Current exploration rate (epsilon): {current_epsilon:.3f}")
                new_epsilon = input(f"Enter new epsilon (0-1, default: keep current): ").strip()
                if new_epsilon:
                    try:
                        game.agent.set_epsilon(float(new_epsilon))
                        print(f"Epsilon set to: {float(new_epsilon):.3f}")
                    except:
                        print("Keeping current epsilon value")
            else:
                print("⚠️  No existing model found. Starting new training...")
            
            episodes = int(input(f"Enter number of additional training episodes (default {TRAINING_EPISODES}): ") or TRAINING_EPISODES)
            game.train(episodes)
            game.agent.save_model("car_model.h5")
            print("✅ Model saved as car_model.h5")
            
        elif choice == "4":
            # CONTINUE training existing model - With visualization
            print("\n🎮 CONTINUING training of existing model with Visualization...")
            print("Press ESC or close window to stop training early")
            game = Game(fast_training=False)
            
            # Try to load existing model
            if game.agent.load_model("car_model.h5"):
                print("✅ Continuing from existing model!")
                # Optionally adjust epsilon for continued exploration
                current_epsilon = game.agent.get_epsilon()
                print(f"Current exploration rate (epsilon): {current_epsilon:.3f}")
                new_epsilon = input(f"Enter new epsilon (0-1, default: keep current): ").strip()
                if new_epsilon:
                    try:
                        game.agent.set_epsilon(float(new_epsilon))
                        print(f"Epsilon set to: {float(new_epsilon):.3f}")
                    except:
                        print("Keeping current epsilon value")
            else:
                print("⚠️  No existing model found. Starting new training...")
            
            episodes = int(input(f"Enter number of additional training episodes (default {TRAINING_EPISODES}): ") or TRAINING_EPISODES)
            game.train(episodes)
            game.agent.save_model("car_model.h5")
            print("✅ Model saved as car_model.h5")
            
        elif choice == "5":
            # Test trained agent
            print("\n🧪 Testing trained agent...")
            print("Press ESC or close window to stop testing early")
            game = Game(fast_training=False)
            try:
                if game.agent.load_model("car_model.h5"):
                    episodes = int(input("Enter number of test episodes (default 10): ") or "10")
                    game.test(episodes)
                else:
                    print("❌ No trained model found. Please train the agent first!")
            except Exception as e:
                print(f"❌ Could not load model: {e}")
                print("Please train the agent first!")
                
        elif choice == "6":
            # Train new model and then test
            print("\n🚀 Starting NEW MODEL training...")
            game = Game(fast_training=True)
            train_episodes = int(input(f"Enter number of training episodes (default {TRAINING_EPISODES}): ") or TRAINING_EPISODES)
            game.train(train_episodes)
            game.agent.save_model("car_model.h5")
            
            print("\n🧪 Testing trained agent...")
            print("Press ESC or close window to stop testing early")
            # Create new game instance for testing with visualization
            game = Game(fast_training=False)
            game.agent.load_model("car_model.h5")
            test_episodes = int(input("Enter number of test episodes (default 10): ") or "10")
            game.test(test_episodes)
            
        else:
            print("❌ Invalid choice!")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up pygame resources
        if game:
            try:
                game.cleanup()
            except:
                pass
        print("👋 Goodbye!")

if __name__ == "__main__":
    # Set TensorFlow logging level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow warnings
    main()

