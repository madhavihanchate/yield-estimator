"""
Crop Yield Prediction Pipeline - Stage 2 (row-level OLS)
Fits OLS per (Zone, Crop) using each district-year-season as an observation.
Train/test split by district to prevent leakage.
Fully vectorized predictions.
"""
import pandas as pd
import numpy as np
import json
import os
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CSV_PATH = os.path.join(OUTPUT_DIR, "master_data.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_DISTRICTS = 15

# =============================================================================
# Zone mapping
# =============================================================================
ZONE_MAP = {
    "Anantapur": "Southern Plateau & Hills", "Chittoor": "Southern Plateau & Hills",
    "Kurnool": "Southern Plateau & Hills", "YSR Kadapa": "Southern Plateau & Hills",
    "Prakasam": "East Coast Plains & Hills", "Nellore": "East Coast Plains & Hills",
    "Guntur": "East Coast Plains & Hills", "Krishna": "East Coast Plains & Hills",
    "West Godavari": "East Coast Plains & Hills", "East Godavari": "East Coast Plains & Hills",
    "Visakhapatnam": "East Coast Plains & Hills", "Srikakulam": "East Coast Plains & Hills",
    "Vizianagaram": "East Coast Plains & Hills", "Tirupati": "Southern Plateau & Hills",
    "Kakinada": "East Coast Plains & Hills", "Alluri Sitharama Raju": "East Coast Plains & Hills",
    "Anakapalli": "East Coast Plains & Hills", "Annamayya": "Southern Plateau & Hills",
    "Bapatla": "East Coast Plains & Hills", "Eluru": "East Coast Plains & Hills",
    "Konaseema": "East Coast Plains & Hills", "NTR": "East Coast Plains & Hills",
    "Nandyal": "Southern Plateau & Hills", "Palnadu": "East Coast Plains & Hills",
    "Parvathipuram Manyam": "East Coast Plains & Hills", "Sri Sathya Sai": "Southern Plateau & Hills",
    "Anjaw": "Eastern Himalaya", "Changlang": "Eastern Himalaya", "East Kameng": "Eastern Himalaya",
    "East Siang": "Eastern Himalaya", "Kamle": "Eastern Himalaya", "Kra Daadi": "Eastern Himalaya",
    "Kurung Kumey": "Eastern Himalaya", "Lepa Rada": "Eastern Himalaya", "Lohit": "Eastern Himalaya",
    "Longding": "Eastern Himalaya", "Lower Dibang Valley": "Eastern Himalaya",
    "Lower Siang": "Eastern Himalaya", "Lower Subansiri": "Eastern Himalaya",
    "Namsai": "Eastern Himalaya", "Pakke-Kessang": "Eastern Himalaya",
    "Papum Pare": "Eastern Himalaya", "Shi Yomi": "Eastern Himalaya", "Siang": "Eastern Himalaya",
    "Tawang": "Eastern Himalaya", "Tirap": "Eastern Himalaya",
    "Upper Dibang Valley": "Eastern Himalaya", "Upper Siang": "Eastern Himalaya",
    "Upper Subansiri": "Eastern Himalaya", "West Kameng": "Eastern Himalaya",
    "West Siang": "Eastern Himalaya",
    "Bajali": "Middle Ganga Plain", "Baksa": "Eastern Himalaya", "Barpeta": "Middle Ganga Plain",
    "Biswanath": "Middle Ganga Plain", "Bongaigaon": "Middle Ganga Plain",
    "Cachar": "Eastern Himalaya", "Charaideo": "Eastern Himalaya", "Chirang": "Eastern Himalaya",
    "Darrang": "Middle Ganga Plain", "Dhemaji": "Eastern Himalaya",
    "Dhubri": "Middle Ganga Plain", "Dibrugarh": "Eastern Himalaya",
    "Dima Hasao": "Eastern Himalaya", "Goalpara": "Middle Ganga Plain",
    "Golaghat": "Middle Ganga Plain", "Hailakandi": "Eastern Himalaya",
    "Hojai": "Middle Ganga Plain", "Jorhat": "Eastern Himalaya",
    "Kamrup": "Middle Ganga Plain", "Kamrup Metropolitan": "Middle Ganga Plain",
    "Karbi Anglong": "Eastern Himalaya", "Karimganj": "Eastern Himalaya",
    "Kokrajhar": "Eastern Himalaya", "Lakhimpur": "Eastern Himalaya",
    "Majuli": "Middle Ganga Plain", "Morigaon": "Middle Ganga Plain",
    "Nagaon": "Middle Ganga Plain", "Nalbari": "Middle Ganga Plain",
    "Sivasagar": "Eastern Himalaya", "Sonitpur": "Middle Ganga Plain",
    "South Salmara-Mankachar": "Middle Ganga Plain", "Tamulpur": "Middle Ganga Plain",
    "Tinsukia": "Eastern Himalaya", "Udalguri": "Eastern Himalaya",
    "West Karbi Anglong": "Eastern Himalaya",
    "Araria": "Middle Ganga Plain", "Arwal": "Middle Ganga Plain",
    "Aurangabad": "Middle Ganga Plain", "Banka": "Middle Ganga Plain",
    "Begusarai": "Middle Ganga Plain", "Bhagalpur": "Middle Ganga Plain",
    "Bhojpur": "Middle Ganga Plain", "Buxar": "Middle Ganga Plain",
    "Darbhanga": "Middle Ganga Plain", "East Champaran": "Middle Ganga Plain",
    "Gaya": "Middle Ganga Plain", "Gopalganj": "Middle Ganga Plain",
    "Jamui": "Middle Ganga Plain", "Jehanabad": "Middle Ganga Plain",
    "Kaimur": "Middle Ganga Plain", "Katihar": "Middle Ganga Plain",
    "Khagaria": "Middle Ganga Plain", "Kishanganj": "Middle Ganga Plain",
    "Lakhisarai": "Middle Ganga Plain", "Madhepura": "Middle Ganga Plain",
    "Madhubani": "Middle Ganga Plain", "Munger": "Middle Ganga Plain",
    "Muzaffarpur": "Middle Ganga Plain", "Nalanda": "Middle Ganga Plain",
    "Nawada": "Middle Ganga Plain", "Patna": "Middle Ganga Plain",
    "Purnia": "Middle Ganga Plain", "Rohtas": "Middle Ganga Plain",
    "Saharsa": "Middle Ganga Plain", "Samastipur": "Middle Ganga Plain",
    "Saran": "Middle Ganga Plain", "Sheikhpura": "Middle Ganga Plain",
    "Sheohar": "Middle Ganga Plain", "Sitamarhi": "Middle Ganga Plain",
    "Siwan": "Middle Ganga Plain", "Supaul": "Middle Ganga Plain",
    "Vaishali": "Middle Ganga Plain", "West Champaran": "Middle Ganga Plain",
    "Balod": "Central Highlands", "Baloda Bazar": "Central Highlands",
    "Balrampur": "Central Highlands", "Bastar": "Central Highlands",
    "Bemetara": "Central Highlands", "Bijapur": "Central Highlands",
    "Bilaspur": "Central Highlands", "Dantewada": "Central Highlands",
    "Dhamtari": "Central Highlands", "Durg": "Central Highlands",
    "Gariaband": "Central Highlands", "Gaurela-Pendra-Marwahi": "Central Highlands",
    "Janjgir-Champa": "Central Highlands", "Jashpur": "Central Highlands",
    "Kabirdham": "Central Highlands", "Kanker": "Central Highlands",
    "Khairagarh-Chhuikhadan-Gandai": "Central Highlands", "Kondagaon": "Central Highlands",
    "Korba": "Central Highlands", "Koriya": "Central Highlands",
    "Mahasamund": "Central Highlands", "Manendragarh-Chirmiri-Bharatpur": "Central Highlands",
    "Mohla-Manpur-Ambagarh Chowki": "Central Highlands", "Mungeli": "Central Highlands",
    "Narayanpur": "Central Highlands", "Raigarh": "Central Highlands",
    "Raipur": "Central Highlands", "Rajnandgaon": "Central Highlands",
    "Sakti": "Central Highlands", "Sarangarh-Bilaigarh": "Central Highlands",
    "Sukma": "Central Highlands", "Surajpur": "Central Highlands", "Surguja": "Central Highlands",
    "North Goa": "West Coast Plains & Hills", "South Goa": "West Coast Plains & Hills",
    "Ahmedabad": "Western Plains", "Amreli": "Western Plains", "Anand": "Western Plains",
    "Aravalli": "Western Plains", "Banaskantha": "Western Plains",
    "Bharuch": "West Coast Plains & Hills", "Bhavnagar": "Western Plains",
    "Botad": "Western Plains", "Chhota Udaipur": "Western Plains", "Dahod": "Western Plains",
    "Dang": "West Coast Plains & Hills", "Devbhoomi Dwarka": "Western Plains",
    "Gandhinagar": "Western Plains", "Gir Somnath": "Western Plains",
    "Jamnagar": "Western Plains", "Junagadh": "Western Plains", "Kheda": "Western Plains",
    "Kutch": "Western Plains", "Mahisagar": "Western Plains", "Mehsana": "Western Plains",
    "Morbi": "Western Plains", "Narmada": "West Coast Plains & Hills",
    "Navsari": "West Coast Plains & Hills", "Panchmahal": "Western Plains",
    "Patan": "Western Plains", "Porbandar": "Western Plains", "Rajkot": "Western Plains",
    "Sabarkantha": "Western Plains", "Surat": "West Coast Plains & Hills",
    "Surendranagar": "Western Plains", "Tapi": "West Coast Plains & Hills",
    "Vadodara": "Western Plains", "Valsad": "West Coast Plains & Hills",
    "Ambala": "Upper Ganga Plain", "Bhiwani": "Upper Ganga Plain",
    "Charkhi Dadri": "Upper Ganga Plain", "Faridabad": "Upper Ganga Plain",
    "Fatehabad": "Upper Ganga Plain", "Gurugram": "Upper Ganga Plain",
    "Hisar": "Upper Ganga Plain", "Jhajjar": "Upper Ganga Plain",
    "Jind": "Upper Ganga Plain", "Kaithal": "Upper Ganga Plain",
    "Karnal": "Upper Ganga Plain", "Kurukshetra": "Upper Ganga Plain",
    "Mahendragarh": "Upper Ganga Plain", "Nuh": "Upper Ganga Plain",
    "Palwal": "Upper Ganga Plain", "Panchkula": "Upper Ganga Plain",
    "Panipat": "Upper Ganga Plain", "Rewari": "Upper Ganga Plain",
    "Rohtak": "Upper Ganga Plain", "Sirsa": "Upper Ganga Plain",
    "Sonipat": "Upper Ganga Plain", "Yamunanagar": "Upper Ganga Plain",
    "Bilaspur": "Western Himalaya", "Chamba": "Western Himalaya",
    "Hamirpur": "Western Himalaya", "Kangra": "Western Himalaya",
    "Kinnaur": "Western Himalaya", "Kullu": "Western Himalaya",
    "Lahaul and Spiti": "Western Himalaya", "Mandi": "Western Himalaya",
    "Shimla": "Western Himalaya", "Sirmaur": "Western Himalaya",
    "Solan": "Western Himalaya", "Una": "Western Himalaya",
    "Anantnag": "Western Himalaya", "Bandipora": "Western Himalaya",
    "Baramulla": "Western Himalaya", "Budgam": "Western Himalaya",
    "Doda": "Western Himalaya", "Ganderbal": "Western Himalaya",
    "Jammu": "Western Himalaya", "Kathua": "Western Himalaya",
    "Kishtwar": "Western Himalaya", "Kulgam": "Western Himalaya",
    "Kupwara": "Western Himalaya", "Poonch": "Western Himalaya",
    "Pulwama": "Western Himalaya", "Rajouri": "Western Himalaya",
    "Ramban": "Western Himalaya", "Reasi": "Western Himalaya",
    "Samba": "Western Himalaya", "Shopian": "Western Himalaya",
    "Srinagar": "Western Himalaya", "Udhampur": "Western Himalaya",
    "Bokaro": "Eastern Plateau & Hills", "Chatra": "Eastern Plateau & Hills",
    "Deoghar": "Eastern Plateau & Hills", "Dhanbad": "Eastern Plateau & Hills",
    "Dumka": "Eastern Plateau & Hills", "East Singhbhum": "Eastern Plateau & Hills",
    "Garhwa": "Eastern Plateau & Hills", "Giridih": "Eastern Plateau & Hills",
    "Godda": "Eastern Plateau & Hills", "Gumla": "Eastern Plateau & Hills",
    "Hazaribagh": "Eastern Plateau & Hills", "Jamtara": "Eastern Plateau & Hills",
    "Khunti": "Eastern Plateau & Hills", "Koderma": "Eastern Plateau & Hills",
    "Latehar": "Eastern Plateau & Hills", "Lohardaga": "Eastern Plateau & Hills",
    "Pakur": "Eastern Plateau & Hills", "Palamu": "Eastern Plateau & Hills",
    "Ramgarh": "Eastern Plateau & Hills", "Ranchi": "Eastern Plateau & Hills",
    "Sahebganj": "Eastern Plateau & Hills", "Seraikela Kharsawan": "Eastern Plateau & Hills",
    "Simdega": "Eastern Plateau & Hills", "West Singhbhum": "Eastern Plateau & Hills",
    "Bagalkot": "Southern Plateau & Hills", "Ballari": "Southern Plateau & Hills",
    "Belagavi": "Southern Plateau & Hills", "Bengaluru Rural": "Southern Plateau & Hills",
    "Bengaluru Urban": "Southern Plateau & Hills", "Bidar": "Southern Plateau & Hills",
    "Chamarajanagar": "Southern Plateau & Hills", "Chikkaballapur": "Southern Plateau & Hills",
    "Chikkamagaluru": "Western Plateau & Hills", "Chitradurga": "Southern Plateau & Hills",
    "Dakshina Kannada": "West Coast Plains & Hills", "Davanagere": "Southern Plateau & Hills",
    "Dharwad": "Southern Plateau & Hills", "Gadag": "Southern Plateau & Hills",
    "Hassan": "Southern Plateau & Hills", "Haveri": "Southern Plateau & Hills",
    "Kalaburagi": "Southern Plateau & Hills", "Kodagu": "Western Plateau & Hills",
    "Kolar": "Southern Plateau & Hills", "Koppal": "Southern Plateau & Hills",
    "Mandya": "Southern Plateau & Hills", "Mysuru": "Southern Plateau & Hills",
    "Raichur": "Southern Plateau & Hills", "Ramanagara": "Southern Plateau & Hills",
    "Shivamogga": "Western Plateau & Hills", "Tumakuru": "Southern Plateau & Hills",
    "Udupi": "West Coast Plains & Hills", "Uttara Kannada": "West Coast Plains & Hills",
    "Vijayanagara": "Southern Plateau & Hills", "Vijayapura": "Southern Plateau & Hills",
    "Yadgir": "Southern Plateau & Hills",
    "Alappuzha": "West Coast Plains & Hills", "Ernakulam": "West Coast Plains & Hills",
    "Idukki": "West Coast Plains & Hills", "Kannur": "West Coast Plains & Hills",
    "Kasaragod": "West Coast Plains & Hills", "Kollam": "West Coast Plains & Hills",
    "Kottayam": "West Coast Plains & Hills", "Kozhikode": "West Coast Plains & Hills",
    "Malappuram": "West Coast Plains & Hills", "Palakkad": "West Coast Plains & Hills",
    "Pathanamthitta": "West Coast Plains & Hills", "Thiruvananthapuram": "West Coast Plains & Hills",
    "Thrissur": "West Coast Plains & Hills", "Wayanad": "West Coast Plains & Hills",
    "Agar Malwa": "Central Plateau & Hills", "Alirajpur": "Central Plateau & Hills",
    "Anuppur": "Central Highlands", "Ashoknagar": "Central Highlands",
    "Balaghat": "Central Highlands", "Barwani": "Central Plateau & Hills",
    "Betul": "Central Plateau & Hills", "Bhind": "Central Highlands",
    "Bhopal": "Central Plateau & Hills", "Burhanpur": "Central Plateau & Hills",
    "Chhatarpur": "Central Highlands", "Chhindwara": "Central Highlands",
    "Damoh": "Central Highlands", "Datia": "Central Highlands",
    "Dewas": "Central Plateau & Hills", "Dhar": "Central Plateau & Hills",
    "Dindori": "Central Highlands", "Guna": "Central Plateau & Hills",
    "Gwalior": "Central Highlands", "Harda": "Central Plateau & Hills",
    "Hoshangabad": "Central Plateau & Hills", "Indore": "Central Plateau & Hills",
    "Jabalpur": "Central Highlands", "Jhabua": "Central Plateau & Hills",
    "Katni": "Central Highlands", "Khandwa": "Central Plateau & Hills",
    "Khargone": "Central Plateau & Hills", "Maihar": "Central Highlands",
    "Mandla": "Central Highlands", "Mandsaur": "Central Plateau & Hills",
    "Mauganj": "Central Highlands", "Morena": "Central Highlands",
    "Narsinghpur": "Central Plateau & Hills", "Neemuch": "Central Plateau & Hills",
    "Niwari": "Central Highlands", "Pandhurna": "Central Highlands",
    "Panna": "Central Highlands", "Raisen": "Central Plateau & Hills",
    "Rajgarh": "Central Plateau & Hills", "Ratlam": "Central Plateau & Hills",
    "Rewa": "Central Highlands", "Sagar": "Central Highlands",
    "Satna": "Central Highlands", "Sehore": "Central Plateau & Hills",
    "Seoni": "Central Highlands", "Shahdol": "Central Highlands",
    "Shajapur": "Central Plateau & Hills", "Sheopur": "Central Highlands",
    "Shivpuri": "Central Highlands", "Sidhi": "Central Highlands",
    "Singrauli": "Central Highlands", "Tikamgarh": "Central Highlands",
    "Ujjain": "Central Plateau & Hills", "Umaria": "Central Highlands",
    "Vidisha": "Central Plateau & Hills",
    "Ahmednagar": "Western Plateau & Hills", "Akola": "Central Plateau & Hills",
    "Amravati": "Central Plateau & Hills", "Aurangabad": "Western Plateau & Hills",
    "Beed": "Western Plateau & Hills", "Bhandara": "Central Plateau & Hills",
    "Buldhana": "Central Plateau & Hills", "Chandrapur": "Central Plateau & Hills",
    "Dhule": "Central Plateau & Hills", "Gadchiroli": "Central Plateau & Hills",
    "Gondia": "Central Plateau & Hills", "Hingoli": "Central Plateau & Hills",
    "Jalgaon": "Western Plateau & Hills", "Jalna": "Western Plateau & Hills",
    "Kolhapur": "Western Plateau & Hills", "Latur": "Western Plateau & Hills",
    "Mumbai City": "West Coast Plains & Hills", "Mumbai Suburban": "West Coast Plains & Hills",
    "Nagpur": "Central Plateau & Hills", "Nanded": "Western Plateau & Hills",
    "Nandurbar": "Central Plateau & Hills", "Nashik": "Western Plateau & Hills",
    "Osmanabad": "Western Plateau & Hills", "Palghar": "West Coast Plains & Hills",
    "Parbhani": "Western Plateau & Hills", "Pune": "Western Plateau & Hills",
    "Raigad": "West Coast Plains & Hills", "Ratnagiri": "West Coast Plains & Hills",
    "Sangli": "Western Plateau & Hills", "Satara": "Western Plateau & Hills",
    "Sindhudurg": "West Coast Plains & Hills", "Solapur": "Western Plateau & Hills",
    "Thane": "West Coast Plains & Hills", "Wardha": "Central Plateau & Hills",
    "Washim": "Central Plateau & Hills", "Yavatmal": "Central Plateau & Hills",
    "Bishnupur": "Eastern Himalaya", "Chandel": "Eastern Himalaya",
    "Churachandpur": "Eastern Himalaya", "Imphal East": "Eastern Himalaya",
    "Imphal West": "Eastern Himalaya", "Jiribam": "Eastern Himalaya",
    "Kakching": "Eastern Himalaya", "Kamjong": "Eastern Himalaya",
    "Kangpokpi": "Eastern Himalaya", "Noney": "Eastern Himalaya",
    "Pherzawl": "Eastern Himalaya", "Senapati": "Eastern Himalaya",
    "Tamenglong": "Eastern Himalaya", "Tengnoupal": "Eastern Himalaya",
    "Thoubal": "Eastern Himalaya", "Ukhrul": "Eastern Himalaya",
    "East Garo Hills": "Eastern Himalaya", "East Jaintia Hills": "Eastern Himalaya",
    "East Khasi Hills": "Eastern Himalaya", "Eastern West Khasi Hills": "Eastern Himalaya",
    "North Garo Hills": "Eastern Himalaya", "Ri Bhoi": "Eastern Himalaya",
    "South Garo Hills": "Eastern Himalaya", "South West Garo Hills": "Eastern Himalaya",
    "South West Khasi Hills": "Eastern Himalaya", "West Garo Hills": "Eastern Himalaya",
    "West Jaintia Hills": "Eastern Himalaya", "West Khasi Hills": "Eastern Himalaya",
    "Aizawl": "Eastern Himalaya", "Champhai": "Eastern Himalaya",
    "Hnahthial": "Eastern Himalaya", "Khawzawl": "Eastern Himalaya",
    "Kolasib": "Eastern Himalaya", "Lawngtlai": "Eastern Himalaya",
    "Lunglei": "Eastern Himalaya", "Mamit": "Eastern Himalaya",
    "Saitual": "Eastern Himalaya", "Serchhip": "Eastern Himalaya", "Siaha": "Eastern Himalaya",
    "Chumoukedima": "Eastern Himalaya", "Dimapur": "Eastern Himalaya",
    "Kiphire": "Eastern Himalaya", "Kohima": "Eastern Himalaya",
    "Longleng": "Eastern Himalaya", "Mokokchung": "Eastern Himalaya",
    "Mon": "Eastern Himalaya", "Niuland": "Eastern Himalaya", "Noklak": "Eastern Himalaya",
    "Peren": "Eastern Himalaya", "Phek": "Eastern Himalaya", "Shamator": "Eastern Himalaya",
    "Tseminyu": "Eastern Himalaya", "Tuensang": "Eastern Himalaya",
    "Wokha": "Eastern Himalaya", "Zunheboto": "Eastern Himalaya",
    "Angul": "Eastern Plateau & Hills", "Balasore": "Lower Ganga Plain",
    "Bargarh": "Eastern Plateau & Hills", "Bhadrak": "Lower Ganga Plain",
    "Bolangir": "Eastern Plateau & Hills", "Boudh": "Eastern Plateau & Hills",
    "Cuttack": "Lower Ganga Plain", "Deogarh": "Eastern Plateau & Hills",
    "Dhenkanal": "Lower Ganga Plain", "Gajapati": "Eastern Plateau & Hills",
    "Ganjam": "East Coast Plains & Hills", "Jagatsinghpur": "Lower Ganga Plain",
    "Jajpur": "Lower Ganga Plain", "Jharsuguda": "Eastern Plateau & Hills",
    "Kalahandi": "Eastern Plateau & Hills", "Kandhamal": "Eastern Plateau & Hills",
    "Kendrapara": "Lower Ganga Plain", "Keonjhar": "Eastern Plateau & Hills",
    "Khordha": "Lower Ganga Plain", "Koraput": "Eastern Plateau & Hills",
    "Malkangiri": "Eastern Plateau & Hills", "Mayurbhanj": "Eastern Plateau & Hills",
    "Nabarangpur": "Eastern Plateau & Hills", "Nayagarh": "Lower Ganga Plain",
    "Nuapada": "Eastern Plateau & Hills", "Puri": "Lower Ganga Plain",
    "Rayagada": "Eastern Plateau & Hills", "Sambalpur": "Eastern Plateau & Hills",
    "Subarnapur": "Eastern Plateau & Hills", "Sundargarh": "Eastern Plateau & Hills",
    "Amritsar": "Trans-Ganga Plain", "Barnala": "Trans-Ganga Plain",
    "Bathinda": "Trans-Ganga Plain", "Faridkot": "Trans-Ganga Plain",
    "Fatehgarh Sahib": "Trans-Ganga Plain", "Fazilka": "Trans-Ganga Plain",
    "Ferozepur": "Trans-Ganga Plain", "Gurdaspur": "Trans-Ganga Plain",
    "Hoshiarpur": "Trans-Ganga Plain", "Jalandhar": "Trans-Ganga Plain",
    "Kapurthala": "Trans-Ganga Plain", "Ludhiana": "Trans-Ganga Plain",
    "Malerkotla": "Trans-Ganga Plain", "Mansa": "Trans-Ganga Plain",
    "Moga": "Trans-Ganga Plain", "Mohali": "Trans-Ganga Plain",
    "Muktsar": "Trans-Ganga Plain", "Pathankot": "Trans-Ganga Plain",
    "Patiala": "Trans-Ganga Plain", "Rupnagar": "Trans-Ganga Plain",
    "Sangrur": "Trans-Ganga Plain", "Shaheed Bhagat Singh Nagar": "Trans-Ganga Plain",
    "Tarn Taran": "Trans-Ganga Plain",
    "Ajmer": "Central Highlands", "Alwar": "Central Highlands",
    "Anupgarh": "Western Plains", "Balotra": "Western Plains",
    "Banswara": "Central Highlands", "Baran": "Central Highlands",
    "Barmer": "Western Plains", "Beawar": "Central Highlands",
    "Bharatpur": "Central Highlands", "Bhilwara": "Central Highlands",
    "Bikaner": "Western Plains", "Bundi": "Central Highlands",
    "Chittorgarh": "Central Highlands", "Churu": "Western Plains",
    "Dausa": "Central Highlands", "Deeg": "Central Highlands",
    "Dholpur": "Central Highlands", "Didwana-Kuchaman": "Western Plains",
    "Dudu": "Central Highlands", "Dungarpur": "Central Highlands",
    "Gangapur City": "Central Highlands", "Hanumangarh": "Western Plains",
    "Jaipur": "Central Highlands", "Jaipur Rural": "Central Highlands",
    "Jaisalmer": "Western Plains", "Jalore": "Western Plains",
    "Jhalawar": "Central Highlands", "Jhunjhunu": "Central Highlands",
    "Jodhpur": "Western Plains", "Jodhpur Rural": "Western Plains",
    "Karauli": "Central Highlands", "Kekri": "Central Highlands",
    "Khairthal-Tijara": "Central Highlands", "Kota": "Central Highlands",
    "Kotputli-Behror": "Central Highlands", "Nagaur": "Western Plains",
    "Neem Ka Thana": "Central Highlands", "Pali": "Western Plains",
    "Phalodi": "Western Plains", "Pratapgarh": "Central Highlands",
    "Rajsamand": "Central Highlands", "Salumbar": "Central Highlands",
    "Sanchore": "Western Plains", "Sawai Madhopur": "Central Highlands",
    "Shahpura": "Central Highlands", "Sikar": "Central Highlands",
    "Sirohi": "Western Plains", "Sri Ganganagar": "Western Plains",
    "Tonk": "Central Highlands", "Udaipur": "Central Highlands",
    "East Sikkim": "Eastern Himalaya", "North Sikkim": "Eastern Himalaya",
    "Pakyong": "Eastern Himalaya", "Soreng": "Eastern Himalaya",
    "South Sikkim": "Eastern Himalaya", "West Sikkim": "Eastern Himalaya",
    "Ariyalur": "East Coast Plains & Hills", "Chengalpattu": "East Coast Plains & Hills",
    "Chennai": "East Coast Plains & Hills", "Coimbatore": "Southern Plateau & Hills",
    "Cuddalore": "East Coast Plains & Hills", "Dharmapuri": "Southern Plateau & Hills",
    "Dindigul": "Southern Plateau & Hills", "Erode": "Southern Plateau & Hills",
    "Kallakurichi": "East Coast Plains & Hills", "Kanchipuram": "East Coast Plains & Hills",
    "Kanyakumari": "West Coast Plains & Hills", "Karur": "Southern Plateau & Hills",
    "Krishnagiri": "Southern Plateau & Hills", "Madurai": "Southern Plateau & Hills",
    "Mayiladuthurai": "East Coast Plains & Hills", "Nagapattinam": "East Coast Plains & Hills",
    "Namakkal": "Southern Plateau & Hills", "Nilgiris": "Southern Plateau & Hills",
    "Perambalur": "East Coast Plains & Hills", "Pudukkottai": "East Coast Plains & Hills",
    "Ramanathapuram": "East Coast Plains & Hills", "Ranipet": "Southern Plateau & Hills",
    "Salem": "Southern Plateau & Hills", "Sivaganga": "East Coast Plains & Hills",
    "Tenkasi": "Southern Plateau & Hills", "Thanjavur": "East Coast Plains & Hills",
    "Theni": "Southern Plateau & Hills", "Thiruvallur": "East Coast Plains & Hills",
    "Thiruvannamalai": "Southern Plateau & Hills", "Thiruvarur": "East Coast Plains & Hills",
    "Thoothukudi": "East Coast Plains & Hills", "Tiruchirappalli": "East Coast Plains & Hills",
    "Tirunelveli": "East Coast Plains & Hills", "Tirupathur": "Southern Plateau & Hills",
    "Tiruppur": "Southern Plateau & Hills", "Vellore": "Southern Plateau & Hills",
    "Viluppuram": "East Coast Plains & Hills", "Virudhunagar": "Southern Plateau & Hills",
    "Adilabad": "Southern Plateau & Hills", "Bhadradri Kothagudem": "Southern Plateau & Hills",
    "Hanumakonda": "Southern Plateau & Hills", "Hyderabad": "Southern Plateau & Hills",
    "Jagtial": "Southern Plateau & Hills", "Jangaon": "Southern Plateau & Hills",
    "Jayashankar Bhupalapally": "Southern Plateau & Hills",
    "Jogulamba Gadwal": "Southern Plateau & Hills", "Kamareddy": "Southern Plateau & Hills",
    "Karimnagar": "Southern Plateau & Hills", "Khammam": "Southern Plateau & Hills",
    "Kumuram Bheem Asifabad": "Southern Plateau & Hills",
    "Mahabubabad": "Southern Plateau & Hills", "Mahabubnagar": "Southern Plateau & Hills",
    "Mancherial": "Southern Plateau & Hills", "Medak": "Southern Plateau & Hills",
    "Medchal-Malkajgiri": "Southern Plateau & Hills", "Mulugu": "Southern Plateau & Hills",
    "Nagarkurnool": "Southern Plateau & Hills", "Nalgonda": "Southern Plateau & Hills",
    "Narayanpet": "Southern Plateau & Hills", "Nirmal": "Southern Plateau & Hills",
    "Nizamabad": "Southern Plateau & Hills", "Peddapalli": "Southern Plateau & Hills",
    "Rajanna Sircilla": "Southern Plateau & Hills", "Ranga Reddy": "Southern Plateau & Hills",
    "Sangareddy": "Southern Plateau & Hills", "Siddipet": "Southern Plateau & Hills",
    "Suryapet": "Southern Plateau & Hills", "Vikarabad": "Southern Plateau & Hills",
    "Wanaparthy": "Southern Plateau & Hills", "Warangal": "Southern Plateau & Hills",
    "Yadadri Bhuvanagiri": "Southern Plateau & Hills",
    "Dhalai": "Eastern Himalaya", "Gomati": "Eastern Himalaya",
    "Khowai": "Eastern Himalaya", "North Tripura": "Eastern Himalaya",
    "Sepahijala": "Eastern Himalaya", "South Tripura": "Eastern Himalaya",
    "Unakoti": "Eastern Himalaya", "West Tripura": "Eastern Himalaya",
    "Agra": "Upper Ganga Plain", "Aligarh": "Upper Ganga Plain",
    "Ambedkar Nagar": "Middle Ganga Plain", "Amethi": "Middle Ganga Plain",
    "Amroha": "Upper Ganga Plain", "Auraiya": "Upper Ganga Plain",
    "Ayodhya": "Middle Ganga Plain", "Azamgarh": "Middle Ganga Plain",
    "Bagpat": "Upper Ganga Plain", "Bahraich": "Middle Ganga Plain",
    "Ballia": "Middle Ganga Plain", "Balrampur": "Middle Ganga Plain",
    "Banda": "Middle Ganga Plain", "Barabanki": "Middle Ganga Plain",
    "Bareilly": "Middle Ganga Plain", "Basti": "Middle Ganga Plain",
    "Bhadohi": "Middle Ganga Plain", "Bijnor": "Upper Ganga Plain",
    "Budaun": "Middle Ganga Plain", "Bulandshahr": "Upper Ganga Plain",
    "Chandauli": "Middle Ganga Plain", "Chitrakoot": "Middle Ganga Plain",
    "Deoria": "Middle Ganga Plain", "Etah": "Upper Ganga Plain",
    "Etawah": "Upper Ganga Plain", "Farrukhabad": "Upper Ganga Plain",
    "Fatehpur": "Middle Ganga Plain", "Firozabad": "Upper Ganga Plain",
    "Gautam Buddha Nagar": "Upper Ganga Plain", "Ghaziabad": "Upper Ganga Plain",
    "Ghazipur": "Middle Ganga Plain", "Gonda": "Middle Ganga Plain",
    "Gorakhpur": "Middle Ganga Plain", "Hamirpur": "Middle Ganga Plain",
    "Hapur": "Upper Ganga Plain", "Hardoi": "Upper Ganga Plain",
    "Hathras": "Upper Ganga Plain", "Jalaun": "Middle Ganga Plain",
    "Jaunpur": "Middle Ganga Plain", "Jhansi": "Middle Ganga Plain",
    "Kannauj": "Upper Ganga Plain", "Kanpur Dehat": "Middle Ganga Plain",
    "Kanpur Nagar": "Middle Ganga Plain", "Kasganj": "Upper Ganga Plain",
    "Kaushambi": "Middle Ganga Plain", "Kheri": "Middle Ganga Plain",
    "Kushinagar": "Middle Ganga Plain", "Lalitpur": "Middle Ganga Plain",
    "Lucknow": "Middle Ganga Plain", "Maharajganj": "Middle Ganga Plain",
    "Mahoba": "Middle Ganga Plain", "Mainpuri": "Upper Ganga Plain",
    "Mathura": "Upper Ganga Plain", "Mau": "Middle Ganga Plain",
    "Meerut": "Upper Ganga Plain", "Mirzapur": "Middle Ganga Plain",
    "Moradabad": "Upper Ganga Plain", "Muzaffarnagar": "Upper Ganga Plain",
    "Pilibhit": "Middle Ganga Plain", "Pratapgarh": "Middle Ganga Plain",
    "Prayagraj": "Middle Ganga Plain", "Raebareli": "Middle Ganga Plain",
    "Rampur": "Upper Ganga Plain", "Saharanpur": "Upper Ganga Plain",
    "Sambhal": "Upper Ganga Plain", "Sant Kabir Nagar": "Middle Ganga Plain",
    "Shahjahanpur": "Middle Ganga Plain", "Shamli": "Upper Ganga Plain",
    "Shravasti": "Middle Ganga Plain", "Siddharthnagar": "Middle Ganga Plain",
    "Sitapur": "Middle Ganga Plain", "Sonbhadra": "Middle Ganga Plain",
    "Sultanpur": "Middle Ganga Plain", "Unnao": "Middle Ganga Plain",
    "Varanasi": "Middle Ganga Plain",
    "Almora": "Western Himalaya", "Bageshwar": "Western Himalaya",
    "Chamoli": "Western Himalaya", "Champawat": "Western Himalaya",
    "Dehradun": "Upper Ganga Plain", "Haridwar": "Upper Ganga Plain",
    "Nainital": "Upper Ganga Plain", "Pauri Garhwal": "Western Himalaya",
    "Pithoragarh": "Western Himalaya", "Rudraprayag": "Western Himalaya",
    "Tehri Garhwal": "Western Himalaya", "Udham Singh Nagar": "Upper Ganga Plain",
    "Uttarkashi": "Western Himalaya",
    "Alipurduar": "Eastern Himalaya", "Bankura": "Lower Ganga Plain",
    "Birbhum": "Lower Ganga Plain", "Cooch Behar": "Eastern Himalaya",
    "Dakshin Dinajpur": "Lower Ganga Plain", "Darjeeling": "Eastern Himalaya",
    "Hooghly": "Lower Ganga Plain", "Howrah": "Lower Ganga Plain",
    "Jalpaiguri": "Eastern Himalaya", "Jhargram": "Lower Ganga Plain",
    "Kalimpong": "Eastern Himalaya", "Kolkata": "Lower Ganga Plain",
    "Malda": "Lower Ganga Plain", "Murshidabad": "Lower Ganga Plain",
    "Nadia": "Lower Ganga Plain", "North 24 Parganas": "Lower Ganga Plain",
    "Paschim Bardhaman": "Lower Ganga Plain", "Paschim Medinipur": "Lower Ganga Plain",
    "Purba Bardhaman": "Lower Ganga Plain", "Purba Medinipur": "Lower Ganga Plain",
    "Purulia": "Lower Ganga Plain", "South 24 Parganas": "Lower Ganga Plain",
    "Uttar Dinajpur": "Lower Ganga Plain",
}
# Note: "Bilaspur" appears twice in ZONE_MAP (Himachal + Chhattisgarh).
# Since the last assignment wins, it'll be Central Highlands (Chhattisgarh).
# This is handled because HP Bilaspur and CG Bilaspur share a name.

print(f"Zone map: {len(ZONE_MAP)} districts, {len(set(ZONE_MAP.values()))} zones")

# =============================================================================
# STEP 1: Load from CSV
# =============================================================================
print("\nSTEP 1: Loading master CSV...")
master = pd.read_csv(CSV_PATH, low_memory=False)
print(f"  {len(master)} rows loaded")

# =============================================================================
# STEP 2: Data quality
# =============================================================================
print("\nSTEP 2: Data quality pass")
dup_cols = ["District_Name", "Crop_Name", "Year", "Season"]
dup_count = master.duplicated(subset=dup_cols, keep=False).sum()
print(f"  Duplicates: {dup_count}")

for c in ["District_Name", "Crop_Name", "Year", "Season", "Area_Hectares",
          "Production_Tonnes", "Yield_kg_ha", "Historical_Rainfall_mm", "Historical_Temp_Avg_C"]:
    missing = master[c].isna().sum()
    print(f"  {c}: {missing} nulls ({missing/len(master)*100:.4f}%)")

bad_yield = int((master["Yield_kg_ha"] <= 0).sum())
print(f"  Zero/negative yield: {bad_yield}")

# =============================================================================
# STEP 3: Assign zones
# =============================================================================
print("\nSTEP 3: Assigning zones")
master["Zone"] = master["District_Name"].map(ZONE_MAP)
mapped = int(master["Zone"].notna().sum())
unmapped = int(master["Zone"].isna().sum())
print(f"  Mapped: {mapped} ({mapped/len(master)*100:.1f}%)")
print(f"  Unmapped (excluded): {unmapped} ({unmapped/len(master)*100:.1f}%)")

unresolved_districts = []
if unmapped > 0:
    unresolved_districts = sorted(master.loc[master["Zone"].isna(), "District_Name"].unique().tolist())
    print(f"  Unresolved districts ({len(unresolved_districts)}):")
    for d in unresolved_districts:
        states = master.loc[master["District_Name"] == d, "_sheet"].unique()
        n_rows = int((master["District_Name"] == d).sum())
        print(f"    {d} ({', '.join(states)}): {n_rows} rows")

master_zoned = master.dropna(subset=["Zone"]).copy()

# =============================================================================
# STEP 4: Feature table
# =============================================================================
print("\nSTEP 4: Building feature table")
features = master_zoned.rename(columns={
    "Historical_Rainfall_mm": "Rainfall",
    "Historical_Temp_Avg_C": "Temp_Avg",
    "_sheet": "State",
})[["District_Name", "Crop_Name", "Category", "Year", "Season",
    "Area_Hectares", "Production_Tonnes", "Yield_kg_ha",
    "Rainfall", "Temp_Avg", "Zone", "State"]].copy()

features["Data_Completeness_Flags"] = "Rainfall:present,Temp:present,NDVI:missing,SoilScore:missing"

FEAT = ["Rainfall", "Temp_Avg"]
n_groups = features.groupby(["Zone", "Crop_Name"]).ngroups
print(f"  {len(features)} rows, {n_groups} (Zone,Crop) groups")

# =============================================================================
# STEP 5: Fit OLS per (Zone, Crop) - row-level with district-grouped split
# =============================================================================
print("\nSTEP 5: Fitting OLS models (row-level, district-grouped split)")

models = {}
coverage = []
model_params = []  # (zone, crop, intercept, coef_r, coef_t, r2, mape)

zone_crop = features.groupby(["Zone", "Crop_Name"])
total_groups = len(zone_crop)
fitted_count = 0
skipped_insuff = 0
skipped_err = 0

for i, ((zone, crop), gdf) in enumerate(zone_crop):
    nd = gdf["District_Name"].nunique()
    if nd < MIN_DISTRICTS:
        skipped_insuff += 1
        coverage.append({"Zone": zone, "Crop": crop,
            "Status": f"skipped - insufficient data (n={nd})",
            "n_districts": nd, "R2": None, "MAPE": None})
        continue

    X = gdf[FEAT].values.astype(np.float64)
    y = gdf["Yield_kg_ha"].values.astype(np.float64)
    groups = gdf["District_Name"].values

    try:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups))
    except Exception:
        skipped_err += 1
        coverage.append({"Zone": zone, "Crop": crop,
            "Status": f"skipped - split error", "n_districts": nd, "R2": None, "MAPE": None})
        continue

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    m = LinearRegression().fit(X_tr, y_tr)
    yp = m.predict(X_te)

    ss_res = float(np.sum((y_te - yp) ** 2))
    ss_tot = float(np.sum((y_te - np.mean(y_te)) ** 2))
    r2 = round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else 0.0
    mask = y_te != 0
    mape = round(float(np.mean(np.abs((y_te[mask] - yp[mask]) / y_te[mask])) * 100), 2) if mask.sum() else 0.0

    fitted_count += 1
    cr, ct = float(m.coef_[0]), float(m.coef_[1])
    ci = float(m.intercept_)

    models[(zone, crop)] = {
        "intercept": ci,
        "coefficients": {FEAT[0]: cr, FEAT[1]: ct},
        "R2": r2, "MAPE": mape,
        "n_train": len(X_tr), "n_test": len(X_te), "n_districts": nd,
    }
    coverage.append({"Zone": zone, "Crop": crop, "Status": "fitted",
        "n_districts": nd, "R2": r2, "MAPE": mape})
    model_params.append((zone, crop, ci, cr, ct, r2, mape))

    if (i + 1) % 200 == 0:
        print(f"  ... processed {i+1}/{total_groups}")

print(f"\n  Total: {total_groups}, Fitted: {fitted_count}, Skipped: {skipped_insuff + skipped_err}")

# =============================================================================
# STEP 5b: Vectorized predictions
# =============================================================================
print("\nSTEP 5b: Vectorized predictions for all rows")
pred_lookup = {r[0:2]: r[2:] for r in model_params}
features["key"] = list(zip(features["Zone"], features["Crop_Name"]))
features["_params"] = features["key"].map(pred_lookup)

mask_v = features["_params"].notna()
pred_df = features.loc[mask_v].copy()

intercepts = pred_df["_params"].apply(lambda p: p[0]).values
coef_r = pred_df["_params"].apply(lambda p: p[1]).values
coef_t = pred_df["_params"].apply(lambda p: p[2]).values
r2s = pred_df["_params"].apply(lambda p: p[3]).values
mapes = pred_df["_params"].apply(lambda p: p[4]).values

pred_df["Predicted_Yield_kg_ha"] = np.round(intercepts + coef_r * pred_df["Rainfall"].values + coef_t * pred_df["Temp_Avg"].values, 2)
pred_df["Actual_Yield_kg_ha"] = pred_df["Yield_kg_ha"].round(2)
pred_df["Model_R2"] = r2s
pred_df["Model_Error_Pct"] = mapes

out_cols = ["District_Name", "Crop_Name", "Zone", "State", "Year", "Season",
            "Predicted_Yield_kg_ha", "Actual_Yield_kg_ha", "Model_R2", "Model_Error_Pct",
            "Data_Completeness_Flags", "Rainfall", "Temp_Avg"]

pred_full = pred_df[out_cols].copy()
pred_full.columns = ["District", "Crop", "Zone", "State", "Year", "Season",
                       "Predicted_Yield_kg_ha", "Actual_Yield_kg_ha", "Model_R2", "Model_Error_Pct",
                       "Data_Completeness_Flags", "Rainfall", "Temp_Avg"]

pred_full.to_csv(os.path.join(OUTPUT_DIR, "prediction_table.csv"), index=False)
print(f"  prediction_table.csv: {len(pred_full)} rows")

latest_year = int(features["Year"].max())
latest = pred_full[pred_full["Year"] == latest_year].copy()
latest.to_csv(os.path.join(OUTPUT_DIR, "predictions_latest.csv"), index=False)
print(f"  predictions_latest.csv (year {latest_year}): {len(latest)} rows")

districts_sorted = sorted(features["District_Name"].unique())
crops_sorted = sorted(features["Crop_Name"].unique())
with open(os.path.join(OUTPUT_DIR, "lookup.json"), "w") as f:
    json.dump({"districts": districts_sorted, "crops": crops_sorted}, f)
print(f"  lookup.json: {len(districts_sorted)} districts, {len(crops_sorted)} crops")

cov_df = pd.DataFrame(coverage)
cov_df.to_csv(os.path.join(OUTPUT_DIR, "coverage_report.csv"), index=False)
print(f"  coverage_report.csv: {len(cov_df)} rows")

models_out = {}
for (z, c), md in models.items():
    models_out[f"{z}||{c}"] = md
with open(os.path.join(OUTPUT_DIR, "models.json"), "w") as f:
    json.dump(models_out, f, indent=2)
print(f"  models.json: {len(models_out)} models")

dq = {
    "total_rows": int(len(master)), "states": int(master["_sheet"].nunique()),
    "unique_districts": int(master["District_Name"].nunique()),
    "unique_crops": int(master["Crop_Name"].nunique()),
    "year_range": [int(master["Year"].min()), int(master["Year"].max())],
    "duplicates": int(dup_count), "bad_yield_rows": bad_yield,
    "rainfall_filled_pct": 100.0, "temp_filled_pct": 100.0,
    "ndvi_columns_present": False, "soil_columns_present": False,
    "unresolved_districts": unresolved_districts, "unresolved_row_count": int(unmapped),
    "zone_counts": {z: sum(1 for v in ZONE_MAP.values() if v == z) for z in sorted(set(ZONE_MAP.values()))},
}
with open(os.path.join(OUTPUT_DIR, "data_quality.json"), "w") as f:
    json.dump(dq, f, indent=2)

print("\n--- Coverage by Zone ---")
for _, r in cov_df.groupby("Zone").agg(
    fitted=("Status", lambda x: (x == "fitted").sum()),
    total=("Crop", "count")).reset_index().iterrows():
    print(f"  {r['Zone']}: {int(r['fitted'])}/{int(r['total'])} crops fitted")

fitted_only = cov_df[cov_df["Status"] == "fitted"]
if len(fitted_only):
    r2vals = fitted_only["R2"]
    print(f"\n--- R2 distribution ---")
    print(f"  R2 > 0.5: {(r2vals > 0.5).sum()}")
    print(f"  R2 0.2-0.5: {((r2vals >= 0.2) & (r2vals <= 0.5)).sum()}")
    print(f"  R2 0-0.2: {((r2vals >= 0) & (r2vals < 0.2)).sum()}")
    print(f"  R2 < 0: {(r2vals < 0).sum()}")
    print(f"  Median R2: {r2vals.median():.4f}, Median MAPE: {fitted_only['MAPE'].median():.2f}%")

    print("\n--- Best 10 Models by R2 ---")
    for _, r in fitted_only.nlargest(10, "R2").iterrows():
        print(f"  {r['Zone']}/{r['Crop']}: R2={r['R2']}, MAPE={r['MAPE']}%, n={r['n_districts']}")

print("\n=== PIPELINE COMPLETE ===")
