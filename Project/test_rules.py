from simulation import Simulation
import os

def test_rules():
    input_file = "smaproject2026/input-S1-14.txt"
    for rule in [1, 2, 3, 4]:
        print(f"\n--- Testing Rule {rule} ---")
        sim = Simulation(input_file, 1, 1, rule)
        sim.setWeekSchedule()
        
        # Check ochtend (slots 0-2) en namiddag (slots 16-18) van maandag (day 0)
        # Merk op: slot 16 is 13:00
        for s in [0, 1, 2, 16, 17, 18]:
            slot = sim.weekSchedule[0][s]
            # Enkel printen als het een electief slot is (type 1)
            if slot.slotType == 1:
                print(f"Slot {s}: Type {slot.slotType}, Start {slot.startTime:.2f}, AppTime {slot.appTime:.4f}")
            else:
                print(f"Slot {s}: Type {slot.slotType} (Niet electief), Start {slot.startTime:.2f}")

if __name__ == "__main__":
    test_rules()
