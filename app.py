import email
import imaplib
import re
import smtplib
from email.header import decode_header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional, List

import openai
import uvicorn
from bson import ObjectId
from fastapi import FastAPI, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, validator

from dotenv import load_dotenv
import os

# Charger le fichier .env
load_dotenv()

# Accéder aux variables d'environnement
MONGO_DETAILS = os.getenv("MONGO_DETAILS")
USERNAME = os.getenv("EMAIL_USERNAME")
PASSWORD = os.getenv("EMAIL_PASSWORD")
IMAP_SERVER = os.getenv("IMAP_SERVER")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))  # Par défaut 587
HF_API_KEY = os.getenv("HF_API_KEY")

# Configuration MongoDB
client = AsyncIOMotorClient(MONGO_DETAILS)
database = client.test
email_collection = database.get_collection("emails")

app = FastAPI()

class EmailModel(BaseModel):
    id: Optional[str] = Field(alias="_id")  # Le champ _id est maintenant facultatif
    subject: str
    sender: str
    body: str
    response: Optional[str] = None
    responded: bool = False

    @validator('id', pre=True, always=True)
    def validate_id(cls, v):
        # Si c'est un ObjectId, le convertir en string
        if isinstance(v, ObjectId):
            return str(v)
        return v

# Client Hugging Face pour l'inférence
# cl = InferenceClient(api_key=HF_API_KEY, provider="together")
# cl = genai.Client(api_key="AIzaSyAyIMRMTyoKSTxuzJ6ZXOnMX3_i1opc-ME")

openai.api_key = os.getenv("OPENAI_API_KEY")

async def query_model(message: str) -> str:
    """Envoie le message au modèle OpenAI et récupère la réponse."""
    try:
        # Envoie le message à l'API OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": message}],
            max_tokens=500  # Ajuste les tokens en fonction de la longueur de la réponse
        )

        # Récupère la réponse générée
        return response['choices'][0]['message']['content']

    except Exception as e:
        return f"❌ Erreur lors de la génération de la réponse : {str(e)}"

async def send_response(to_address, subject, body):
    """Envoie une réponse par email."""
    msg = MIMEMultipart()
    msg["From"] = USERNAME
    msg["To"] = to_address
    msg["Subject"] = "Re: " + subject
    msg.attach(MIMEText(body, "plain"))
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(USERNAME, PASSWORD)
        server.sendmail(USERNAME, to_address, msg.as_string())
        server.quit()
    except Exception as e:
        return f"❌ Erreur lors de l'envoi de la réponse : {str(e)}"
    return "Réponse envoyée avec succès"

def extract_email_and_name(sender: str):
    """Extrait l'adresse email et le nom du sender."""
    match = re.match(r"(.+?)\s*<([^>]+)>", sender)
    if match:
        name, email = match.groups()
    else:
        email = sender  # Si pas de format "Nom <email>", stocke tel quel
        name = None  # Nom non disponible
    return name, email


def check_new_mail():
    """Vérifie les nouveaux mails et répond si nécessaire."""
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(USERNAME, PASSWORD)
        mail.select("inbox")

        status, messages = mail.search(None, "UNSEEN")
        mail_ids = messages[0].split()

        if not mail_ids:
            print("✅ Aucun nouveau mail non lu.")
            return None

        latest_mail_id = mail_ids[-1]
        status, msg_data = mail.fetch(latest_mail_id, "(RFC822)")

        if not msg_data or msg_data[0] is None:
            print("⚠️ Impossible de récupérer l'email.")
            return None

        msg = email.message_from_bytes(msg_data[0][1])

        subject, encoding = decode_header(msg["Subject"])[0]
        if isinstance(subject, bytes):
            subject = subject.decode(encoding if encoding else "utf-8")

        sender = msg.get("From")
        sender_name, sender_email = extract_email_and_name(sender)  # Extraction

        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain" and "attachment" not in str(part.get("Content-Disposition")):
                    body = part.get_payload(decode=True).decode(errors="ignore")
                    break
        else:
            if msg.get_content_type() == "text/plain":
                body = msg.get_payload(decode=True).decode(errors="ignore")

        if not body.strip():
            print("❌ Aucun contenu texte détecté.")
            return None

        print(f"📩 Nouveau mail de {sender_email} - Sujet: {subject}")
        print(f"📜 Contenu: {body[:200]}...")

        mail.logout()

        return {
            "subject": subject,
            "sender": sender_email,
            "sender_name": sender_name,  # Ajout du nom séparé
            "body": body,
        }

    except Exception as e:
        print(f"❌ Erreur lors de la vérification des mails : {str(e)}")
        return None


@app.post("/fetch-emails/")
async def fetch_emails():
    """Récupère le dernier mail non lu et l'ajoute dans la base de données."""
    email_data = check_new_mail()

    if not email_data:
        return {"message": "Aucun nouveau mail non lu trouvé."}

    # Ajout d'un ObjectId
    email_data["_id"] = ObjectId()

    # Insertion dans MongoDB
    result = await email_collection.insert_one(email_data)
    email_data["_id"] = str(result.inserted_id)  # Convertir en string pour la réponse API

    return {
        "message": "Email récupéré et ajouté dans la base de données",
        "email": email_data
    }



@app.post("/generate-response/{email_id}")
async def generate_response(email_id: str):
    """Génère une réponse à un email spécifique à partir de son ID et met à jour la base de données."""
    email_data = await email_collection.find_one({"_id": ObjectId(email_id)})

    if not email_data:
        raise HTTPException(status_code=404, detail="Email introuvable")

    response = await query_model(email_data["body"])

    # Mise à jour de l'email dans la base de données pour inclure la réponse générée
    await email_collection.update_one({"_id": ObjectId(email_id)}, {"$set": {"response": response, "responded": True}})

    return {"message": "Réponse générée avec succès", "response": response}

@app.post("/send-response/{email_id}")
async def send_email_response(email_id: str):
    email_data = await email_collection.find_one({"_id": ObjectId(email_id)})
    if not email_data:
        raise HTTPException(status_code=404, detail="Email introuvable")
    if not email_data["response"]:
        raise HTTPException(status_code=400, detail="Aucune réponse générée pour cet email")
    send_status = await send_response(email_data["sender"], email_data["subject"], email_data["response"])
    if "❌" in send_status:  # Gestion des erreurs lors de l'envoi
        raise HTTPException(status_code=500, detail=send_status)
    return {"message": send_status}

@app.get("/emails/", response_model=List[EmailModel])
async def get_emails():
    emails = await email_collection.find().to_list(1000)
    return emails

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur mon API FastAPI déployée!"}

@app.get("/emails/unresponded/", response_model=List[EmailModel])
async def get_unresponded_emails():
    emails = await email_collection.find({"responded": False}).to_list(1000)
    return emails

class ChatMessage(BaseModel):
    message: str

@app.post("/chatbot/")
async def chatbot_response(chat_message: ChatMessage):
    """Prend un message utilisateur, envoie au modèle, et retourne la réponse générée"""
    try:
        response = await query_model(chat_message.message)
        return {"generated_response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération de la réponse : {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
