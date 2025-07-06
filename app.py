from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import matplotlib
import networkx as nx
from flask_socketio import SocketIO, emit, join_room
matplotlib.use('Agg')


from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch

# Path to your model directory
model_path = "Chatbot/trained_chatbot_model"

# Load tokenizer and model
My_tokenizer = AutoTokenizer.from_pretrained(model_path)
Instamodel = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
print("Model loaded successfully!")
chat_history = []



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key'

app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')  # Folder to save uploaded images in 'static/uploads'


if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)


socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
# Database models
# Load YOLOv9 model (ensure 'best.pt' is in the correct directory)
model = YOLO('best.pt')

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    hospital = db.Column(db.String(250), nullable=False)
    appointment_date = db.Column(db.String(50), nullable=False)
    appointment_time = db.Column(db.String(50), nullable=False)
    notes = db.Column(db.Text, nullable=True)

if My_tokenizer.pad_token is None:
    My_tokenizer.pad_token = My_tokenizer.eos_token  # Use eos_token as pad_token


def generate_chatbot_response(user_input):
    formatted_input = f"User: {user_input}\nBot:"
    inputs = My_tokenizer(formatted_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
    attention_mask = inputs["attention_mask"]
    outputs = Instamodel.generate(
        inputs['input_ids'],
        attention_mask=attention_mask, 
        max_length=150, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2, 
        pad_token_id=My_tokenizer.pad_token_id
    )
    response = My_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]  # Proper decoding
    if "Bot:" in response:
        response = response.split("Bot:")[-1].strip()


    return response.strip()  # Remove extra spaces/newlines




with app.app_context():
    db.create_all()


# Helper function to detect fractures and other conditions
# Inside detect_condition function
def detect_condition(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)
    
    # Save the detection image
    detection_image_path = os.path.join(app.config['UPLOAD_FOLDER'], "detection.png")
    plt.figure(figsize=(10, 10))
    plt.imshow(results[0].plot())  # This will plot detections
    plt.axis('off')
    plt.savefig(detection_image_path)  # Save for later display in the template
    plt.close()

    # detected_labels = set([result.te for result in results[0].boxes.data])

    cls_tensor=results[0].boxes.cls
    conf_tensor=results[0].boxes.conf
    result_list = [[cls.item(), conf.item()] for cls, conf in zip(cls_tensor, conf_tensor)]
    cls = results[0].boxes.cls
    conf = results[0].boxes.conf
    xyxy = results[0].boxes.xyxy



        
    # Initialize the graph
    # Initialize the graph
    g = nx.DiGraph()

    # Add nodes for the layers with additional tensor information
    g.add_node("Input", pos=(0, 3), layer="Input Layer", tensor="Shape: (640, 288)")

    num_detections = len(cls)  # Count detected objects

    g.add_node("YOLO", pos=(1, 2), layer="YOLO Detection", tensor=f"cls: {cls[0].tolist()}\nconf: {conf[0].tolist()}")

    if num_detections > 1:
        g.add_node("EfficientDet", pos=(1, 1), layer="EfficientDet Detection", tensor=f"cls: {cls[1].tolist()}\nconf: {conf[1].tolist()}")
        g.add_edge("YOLO", "EfficientDet")  # Only add edge if EfficientDet exists

    if num_detections > 2:
        g.add_node("DART", pos=(1, 0), layer="DART Classification", tensor=f"cls: {cls[2].tolist()}\nconf: {conf[2].tolist()}")
        g.add_edge("EfficientDet", "DART")  # Only add edge if DART exists

    g.add_node("Output", pos=(2, 1), layer="Output Layer", tensor="Output predictions")

    # Add edges to represent flow between layers
    g.add_edge("Input", "YOLO")
    if "EfficientDet" in g:
        g.add_edge("EfficientDet", "Output")  # Only connect if EfficientDet exists
    if "DART" in g:
        g.add_edge("DART", "Output")  # Only connect if DART exists

    # Define positions for visualization
    pos = nx.get_node_attributes(g, 'pos')

    # Ensure that all nodes in the graph have positions
    if set(g.nodes) - set(pos.keys()):
        print(f"Warning: The following nodes have no position: {set(g.nodes) - set(pos.keys())}")

    # Draw the graph only if all nodes have positions
    plt.figure(figsize=(12, 8))
    nx.draw(
        g, 
        pos, 
        with_labels=True, 
        node_size=2000, 
        node_color="lightblue",
        font_size=10
    )

    # Annotate the layers with tensor info
    for node in g.nodes:
        layer_tensor_info = g.nodes[node]['tensor']
        x_pos, y_pos = pos[node]
        plt.text(x_pos, y_pos - 0.2, layer_tensor_info, fontsize=9, ha='center')


    plt.savefig(os.path.join('static', 'hybrid_model_layers_with_tensor.png'))

    names = {
    0: 'boneanomaly', 1: 'bonelesion', 2: 'foreignbody', 3: 'fracture',
    4: ' ', 5: 'periostealreaction', 6: 'pronatorsign', 7: 'softtissue', 8: 'text'
    }
    l=[]
    for tensor in results[0].boxes.data:
        class_id = int(tensor[-1].item())  # Get the last value and convert to an integer
        label = names.get(class_id, "Unknown")  # Retrieve the label from `names`
        if label=='text':
            pass
        else:
            l.append(label)
        print(f"Class ID: {class_id}, Label: {label}")

    with open("results.txt", "w") as file:
        file.write("\n".join(str(item) for item in results[0].boxes.data))  
    return l, detection_image_path, result_list


# Home route
@app.route('/')
def home():
    #return render_template('loginHospital.html')
    return render_template('index.html')


@app.route('/map')
def map_view():
    precision = 0.85  # Example precision value
    recall = 0.80     # Example recall value
    f1 = 0.82         # Example F1 score value
    result_list = session.get('result_list', [])
    return render_template('map.html', result_list=result_list,precision=precision,recall=recall,f1=f1)


# Login route
# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('pswd')
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id  # Store user ID in session
            session['email'] = user.email
            flash("Logged in successfully!", "success")
            return redirect(url_for('home')) 
        else:
            flash("Invalid credentials. Please try again.", "danger")
    return render_template('login.html')

# Signup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')  # Use .get() to avoid KeyError
        email = request.form.get('email')
        password = request.form.get('pswd')
        
        if User.query.filter_by(email=email).first():
            flash("Email already exists. Please choose a different one.", "danger")
        else:
            new_user = User(
                username=username,
                email=email,
                password=generate_password_hash(password, method='pbkdf2:sha256')
            )
            db.session.add(new_user)
            db.session.commit()
            flash("Account created successfully! You can now log in.", "success")
            return redirect(url_for('login'))

    return render_template('login.html')

# Book appointment route
@app.route('/book', methods=['GET', 'POST'])
def book():
    hospital_name = request.args.get('hospital') 
    if request.method == 'POST':
        appointment = Appointment(
            full_name=request.form['fullName'],
            email=request.form['email'],
            phone=request.form['phone'],
            hospital=request.form['hospital'],
            appointment_date=request.form['appointmentDate'],
            appointment_time=request.form['appointmentTime'],
            notes=request.form['notes']
        )
        db.session.add(appointment)
        db.session.commit()
        flash("Appointment request submitted successfully!", "success")
        return redirect(url_for('home'))
    return render_template('Contact.html', hospital=hospital_name)


# Appointments route
@app.route('/appointments')
def appointments():
    if 'user_id' not in session:
        flash("Please log in to view your appointments.", "warning")
        return redirect(url_for('login'))
    
    # Fetch all appointments for the logged-in user
    user_appointments = Appointment.query.filter_by(email=session.get('email')).all()
    print("User is logged in with email:", session.get('email'))
    print("Fetched appointments:", user_appointments)
    return render_template('appointments.html', appointments=user_appointments)

# Cancel appointment route
@app.route('/cancel_appointment/<int:appointment_id>', methods=['POST'])
def cancel_appointment(appointment_id):
    appointment = Appointment.query.get(appointment_id)
    if appointment:
        db.session.delete(appointment)
        db.session.commit()
        flash("Appointment canceled successfully.", "success")
    else:
        flash("Appointment not found.", "danger")
    return redirect(url_for('appointments'))




@socketio.on('send_message')
def handle_message(data):
    """Handles incoming messages from the client."""
    user_message = data.get('message')  # Message sent by the user
    if not user_message:
        emit('receive_message', {'message': 'Message cannot be empty.'})
        return
    
    try:
        # Generate chatbot response
        chatbot_response = generate_chatbot_response(user_message)
        
        # Add messages to the chat history
        chat_history.append({'user': user_message, 'bot': chatbot_response})
        
        # Emit the chatbot response back to the client
        emit('receive_message', {'message': chatbot_response})
    except Exception as e:
        emit('receive_message', {'message': f"An error occurred: {str(e)}"})



# Image Upload and Detection route
@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['image']
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Run detection on the uploaded image
            condition, detection_image_path, result_list = detect_condition(file_path)

            print(result_list)
            session['result_list'] = result_list
            # Check for fracture
            if 'fracture' in condition:
                flash("Fracture detected! Redirecting to nearby hospitals...", "warning")
                return redirect(url_for('map_view'))
            
            # Display homemade remedies for other conditions
            home_remedies = {
                'boneanomaly': "Consider rest and avoid heavy lifting.",
                'bonelesion': "Apply a cold compress to reduce swelling.",
                'foreignbody': "Clean the affected area and monitor for signs of infection.",
                'metal': "No treatment needed, but monitor for pain.",
                'periostealreaction': "Rest and consult a specialist if pain persists.",
                'pronatorsign': "Apply ice and take anti-inflammatory medication if necessary.",
                'softtissue': "Apply ice, elevate the area, and rest.",
                'text': "Follow general first aid procedures."
            }
            
            # Filter remedies based on detected conditions
            remedies = {cond: home_remedies[cond] for cond in condition if cond in home_remedies}
            # return render_template('remedies.html', remedies=remedies)
            return render_template('remedies.html', remedies=remedies, image_path=detection_image_path)
    
    return render_template('upload.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)  # Remove user ID from the Flask session
    flash("You have been logged out.", "success")
    return redirect(url_for('home'))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    socketio.run(app,host='0.0.0.0', port=5000, debug=True)
