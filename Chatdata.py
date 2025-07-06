import pandas as pd
# Additional 50 records for "Fracture Information & Remedies"
fracture_data = [
    ("What should I do if I suspect a fracture?", "Keep the injured area still, apply ice, and seek medical attention."),
    ("Can I treat a simple fracture at home?", "No, fractures require medical evaluation and proper immobilization."),
    ("How is a fracture diagnosed?", "Doctors use X-rays, CT scans, or MRI scans to diagnose fractures."),
    ("Do fractures always need a cast?", "Not always; some minor fractures heal with splints or braces."),
    ("What foods help heal fractures?", "Foods rich in calcium, vitamin D, and protein help bone healing."),
    ("Can a fracture heal without treatment?", "No, untreated fractures can lead to complications and improper healing."),
    ("What’s the difference between a sprain and a fracture?", "A sprain affects ligaments, while a fracture is a broken bone."),
    ("How long does a fracture take to heal?", "A simple fracture takes 6-8 weeks, but it depends on severity and age."),
    ("Can I exercise with a healing fracture?", "Gentle movement may be allowed, but avoid weight-bearing activities."),
    ("What are the signs of a severe fracture?", "Severe pain, deformity, swelling, and inability to move the limb."),
    ("What should I do for a child’s fracture?", "Keep them calm, immobilize the area, and get medical help immediately."),
    ("Can fractures cause permanent damage?", "If untreated or severe, fractures can lead to long-term complications."),
    ("How do doctors treat fractures?", "Treatment includes immobilization with a cast, splint, or sometimes surgery."),
    ("Do fractures hurt while healing?", "Mild discomfort is common, but severe pain should be reported to a doctor."),
    ("Can a fracture get worse?", "Yes, moving the injured area or ignoring treatment can worsen the fracture."),
    ("Is it safe to move someone with a fracture?", "Avoid unnecessary movement to prevent further injury."),
    ("Can fractures heal faster?", "Proper nutrition, rest, and following medical advice can speed up healing."),
    ("What is an open fracture?", "An open fracture occurs when the bone breaks through the skin."),
    ("Can I use painkillers for a fracture?", "Over-the-counter painkillers can help, but consult a doctor for advice."),
    ("What is a stress fracture?", "A stress fracture is a small crack in the bone due to repetitive strain."),
    ("Can fractures cause infections?", "Open fractures can lead to infections if not treated properly."),
    ("What is a hairline fracture?", "A hairline fracture is a small, thin break in the bone."),
    ("Do I need surgery for a fracture?", "Severe or displaced fractures may require surgery for proper healing."),
    ("Can I sleep with a cast on?", "Yes, but keep the cast elevated to reduce swelling."),
    ("What happens if a fracture heals incorrectly?", "Improper healing can cause deformities, pain, and reduced mobility."),
    ("Can I drive with a fractured arm?", "No, it's unsafe to drive with a fractured or immobilized limb."),
    ("How do I prevent fractures?", "Ensure a calcium-rich diet, exercise, and avoid high-risk activities."),
    ("What’s the best way to care for a cast?", "Keep it dry, don’t insert objects inside, and avoid excessive pressure."),
    ("Can a fracture cause numbness?", "Yes, nerve damage from a fracture can cause numbness or tingling."),
    ("What’s the difference between a fracture and a break?", "There is no difference; a fracture is the medical term for a broken bone."),
    ("How do I shower with a cast?", "Use a waterproof cover or wrap the cast in plastic to keep it dry."),
    ("Can fractures occur from osteoporosis?", "Yes, weakened bones from osteoporosis increase fracture risk."),
    ("How can I tell if a fracture is healing?", "Reduced pain and improved mobility indicate healing, but X-rays confirm it."),
    ("Do fractures heal differently in children and adults?", "Children’s bones heal faster and have better remodeling potential."),
    ("Can smoking affect bone healing?", "Yes, smoking slows down bone healing and increases complications."),
    ("What is a greenstick fracture?", "A greenstick fracture is an incomplete break, common in children."),
    ("Why do elderly people have higher fracture risks?", "Aging weakens bones and increases the risk of falls and fractures."),
    ("Can I remove my cast at home?", "No, only a doctor should remove a cast to prevent injury."),
    ("What is a displaced fracture?", "A displaced fracture means the broken bone has moved out of alignment."),
    ("Do fractures itch while healing?", "Yes, itching is common due to skin irritation and cast pressure."),
    ("Can I drink alcohol with a fracture?", "Excessive alcohol can slow bone healing and should be avoided."),
    ("How do I know if my fracture is infected?", "Redness, swelling, pus, and fever may indicate an infection."),
    ("Can fractures be prevented?", "Yes, by maintaining bone health, exercising, and using safety gear."),
    ("How does age affect fracture healing?", "Older individuals take longer to heal due to slower cell regeneration."),
    ("What is the first aid for an open fracture?", "Cover with a clean cloth, immobilize, and seek emergency care."),
    ("Can I work with a fractured arm?", "It depends on the job; consult your doctor about work restrictions."),
    ("Do fractures hurt more at night?", "Pain can feel worse at night due to reduced distractions and movement."),
    ("Can a fracture cause swelling?", "Yes, swelling is a common symptom due to inflammation and tissue damage."),
    ("How do I reduce swelling from a fracture?", "Keep the injured limb elevated and apply ice to reduce swelling."),
]

# Convert the new data into a DataFrame and append it to the existing one
df_new = pd.DataFrame(fracture_data, columns=["User Prompt", "Chatbot Response"])
df=pd.read_csv("main.csv")
df = pd.concat([df, df_new], ignore_index=True)

# Save updated dataset
file_path_updated = "healthcare_chatbot_dataset_updated.csv"
df.to_csv(file_path_updated, index=False)

# Return the new file path
file_path_updated
