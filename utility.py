from myanmartools import ZawgyiDetector
from icu import Transliterator
import re 
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
from tensorflow.keras.models import load_model

detector = ZawgyiDetector()
converter = Transliterator.createInstance('Zawgyi-my')

url = "https://drive.google.com/file/d/1Y1uLXos6Mk-faLqCRIlt93dJisZmNwuF/view?usp=drive_link"
gdown.download(url, "model.h5", quiet=False)
model = tf.keras.models.load_model("model.h5")

# model_path = "Models/syllable_model.keras"  
vectorizer_path = "Tokenizer/syllable_tokenizer.pkl"

# model = load_model(model_path)
tokenizer = joblib.load(vectorizer_path)



def detect_and_convert(text):
    """
    Detect if the text is in Zawgyi or Unicode.
    If it's Zawgyi, convert it to Unicode.
    """
    # Detect the encoding
    zawgyi_probability = detector.get_zawgyi_probability(text)

    # If the probability is greater than 0.5, assume it's Zawgyi
    if zawgyi_probability > 0.5:
        # Convert Zawgyi to Unicode
        unicode_text = converter.zawgyi_to_unicode(text)
        return unicode_text
    else:
        return text



def character_tokenization(text: str) -> str:
    """
    Tokenizes text on character level.
    Returns space-separated characters.
    """
    return ' '.join(list(text))



def syllable_tokenization(input:str)->str:
    #input = re.sub(r"\s", "", input.strip())
    return re.sub(r"(([A-Za-z0-9]+)|[က-အ|ဥ|ဦ](င်္|[က-အ][ှ]*[့း]*[်]|္[က-အ]|[ါ-ှႏꩻ][ꩻ]*){0,}|.)",r"\1 ", input)


def multilingual_semi_syllable_break(user_input):
  # Multilingual Semi Syllabe Break (Lao, Kannada, Oriya, Gujarati, Malayalam, Khmer, Bengali, Sinhala, Tamil, Shan, Mon, Pali and Sanskrit, Sagaw Karen, Western Poh Karen, Eastern Poh Karen, Geba Karen, Kayah, Rumai Palaung, Khamathi Shan, Aiton and Phake, Burmese (Myanmar), Paoh, Rakhine Languages)
  result = re.sub(r"([a-zA-Z]+|[຀-ຯຽ-໇ໜ-ໟ][ະ-ຼ່-໏]{0,}|[಄-಻ೞ-ೡ][಼ಀ-ಃಾ-ೝೢ-೥]{0,}|[ଅ-଻ଡ଼-ୡୱ][଼଀-଄ା-୛ୢ-୥]{0,}|[અ-઻ૐ-૟ૠ-ૡ૰ૹ][઀-઄઼ા-૏-ૣૺ-૿]{0,}|[അ-ഺ൏-ൡ൰-ൿ][ഀ-ഄ഻-഼ാ-ൎൢ-൥]{0,}|[ក-ឳ។-៚ៜ][ា-៓៝]{0,}|[అ-ఽౘ-ౡ౷౸-౿][ఀ-ఄా-౗ౢ-౥]{0,}|[অ-঻ড়-ৡৰ-৽][ঁ-঄়-৛ৢ-৥৾-৿্]{0,}|[අ-෉][්-෥ෲ-ෳ඀-඄ි]{0,}|[அ-஽][஀-஄ா-௏ௗ]{0,}|[က-ဪဿ၌-၏ၐ-ၕၚ-ၝၡၥၦၮ-ၰၵ-ႁႎ႐-႙႟][ါ-ှၖ-ၙၞ-ၠၢ-ၤၧ-ၭၱ-ၴႂ-ႍႏႚ-႞ꩻ]{0,}|.)",r"\1 ", user_input)
  result = re.sub(r" +", " ", result).strip()
  return result


def transform_text(text, max_len=600):
    tokenized_text = syllable_tokenization(text)  
    
    # Then convert to sequence
    sequences = tokenizer.texts_to_sequences([tokenized_text])
    
    # Pad the sequence
    padded = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded
