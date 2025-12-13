import RPi.GPIO as GPIO
import time
import sys

# Using BCM numbering mode 
GPIO_DIR_PIN = 27  # BCM pin for L298N IN1 (Direction)
GPIO_PWM_PIN = 13  # BCM pin for L298N ENA (Speed Control / Enable)

PWM_FREQUENCY = 100  # Hz
INITIAL_SPEED = 25   # % Duty Cycle (0-100)

try:
    # Use BCM numbering mode
    GPIO.setmode(GPIO.BCM)

    # Configure pins as outputs
    GPIO.setup(GPIO_DIR_PIN, GPIO.OUT)
    GPIO.setup(GPIO_PWM_PIN, GPIO.OUT)

    # Initialize PWM on the enable pin
    motor_pwm = GPIO.PWM(GPIO_PWM_PIN, PWM_FREQUENCY)

    # Start the motor with the initial speed
    # The direction (IN1) is set to HIGH for one direction
    GPIO.output(GPIO_DIR_PIN, GPIO.HIGH)
    motor_pwm.start(INITIAL_SPEED)
    
    print(f"Motor running at {INITIAL_SPEED}% speed (Direction HIGH).")
    print("Press Ctrl+C to stop sooner.")

    # Run for a set duration
    time.sleep(3) 
    
    # Stop PWM and cleanup
    motor_pwm.stop()
    print("\nMotor stopped.")

except KeyboardInterrupt:
    print("\nScript stopped by user.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)
finally:
    # 8. Cleanup: Critical step to reset pins on exit
    GPIO.cleanup()
    print("GPIO cleanup complete. Pins reset to default state.")