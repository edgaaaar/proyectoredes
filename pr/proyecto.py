import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import sys

# --- CLASE PRINCIPAL (NO SE TOCÓ) ---
class HammingChannel:
    def __init__(self):
        # Matriz Generadora G (4x7)
        self.G = np.array([
            [1, 0, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1]
        ])

        # Matriz de Paridad H (3x7)
        self.H = np.array([
            [1, 1, 0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 1]
        ])
        
        # Mapeo de Síndrome
        self.syndrome_map = {}
        for i in range(7):
            col = tuple(self.H[:, i])
            self.syndrome_map[col] = i

    def text_to_bits(self, value, num_bits):
        return [int(x) for x in format(value, f'0{num_bits}b')]

    def bits_to_text(self, bits):
        return int("".join(str(x) for x in bits), 2)

    def encode_block(self, nibble_4bits):
        msg_vector = np.array(nibble_4bits)
        encoded = np.dot(msg_vector, self.G) % 2
        return encoded.astype(int)

    def decode_block(self, received_7bits):
        r_vector = np.array(received_7bits)
        syndrome = np.dot(self.H, r_vector) % 2
        syndrome_tuple = tuple(syndrome)
        
        error_detected = False
        corrected_vector = r_vector.copy()

        if np.any(syndrome):
            error_detected = True
            if syndrome_tuple in self.syndrome_map:
                error_pos = self.syndrome_map[syndrome_tuple]
                corrected_vector[error_pos] = 1 - corrected_vector[error_pos]
        
        decoded_data = corrected_vector[:4]
        return decoded_data, error_detected

    def simulate_noise(self, encoded_msg, error_prob=0.1):
        noisy_msg = encoded_msg.copy()
        if random.random() < error_prob:
            bit_to_flip = random.randint(0, 6)
            noisy_msg[bit_to_flip] = 1 - noisy_msg[bit_to_flip]
        return noisy_msg

# --- FUNCIÓN IMAGEN (NO SE TOCÓ) ---
def procesar_imagen(image_path, noise_level=0.2):
    print(f"\n--- Iniciando Simulación Hamming (7,4) con IMAGEN ---")
    print(f"Cargando imagen: {image_path}")
    
    try:
        img = Image.open(image_path).convert('L')
        img = img.resize((100, 100)) 
        img_arr = np.array(img)
        original_shape = img_arr.shape
        flattened_pixels = img_arr.flatten()
    except Exception as e:
        print(f"Error cargando imagen: {e}")
        return

    hamming = HammingChannel()
    
    encoded_stream = []
    noisy_stream = []
    corrected_stream = []
    total_errors = 0

    print("Procesando píxeles...")
    
    for pixel in flattened_pixels:
        bits = hamming.text_to_bits(pixel, 8)
        nibble_high = bits[:4]
        nibble_low = bits[4:]
        
        parts_reconstructed = []
        
        for nibble in [nibble_high, nibble_low]:
            encoded = hamming.encode_block(nibble)
            encoded_stream.append(encoded)
            noisy = hamming.simulate_noise(encoded, error_prob=noise_level)
            noisy_stream.append(noisy)
            decoded_nibble, error_found = hamming.decode_block(noisy)
            if error_found: total_errors += 1
            parts_reconstructed.extend(decoded_nibble)

        corrected_pixel = hamming.bits_to_text(parts_reconstructed)
        corrected_stream.append(corrected_pixel)

    noisy_pixels = []
    for i in range(0, len(noisy_stream), 2):
        n1 = noisy_stream[i][:4]
        n2 = noisy_stream[i+1][:4]
        val = hamming.bits_to_text(np.concatenate((n1, n2)))
        noisy_pixels.append(val)
        
    img_noisy = np.array(noisy_pixels).reshape(original_shape)
    img_corrected = np.array(corrected_stream).reshape(original_shape)

    print(f"Errores corregidos: {total_errors}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.title("Original"); plt.imshow(img_arr, cmap='gray'); plt.axis('off')
    plt.subplot(1, 3, 2); plt.title(f"Con Ruido"); plt.imshow(img_noisy, cmap='gray'); plt.axis('off')
    plt.subplot(1, 3, 3); plt.title("Corregida"); plt.imshow(img_corrected, cmap='gray'); plt.axis('off')
    plt.tight_layout()
    plt.show()

# --- NUEVA FUNCIÓN: PROCESAR CADENA MANUAL ---
def procesar_cadena_manual():
    print(f"\n--- Simulación Manual Hamming (7,4) ---")
    cadena = input("Ingresa una cadena binaria (ej. 10110001): ")
    
    # Validar que solo sean 0s y 1s
    if not all(c in '01' for c in cadena):
        print("Error: La cadena solo puede contener 0s y 1s.")
        return

    # Rellenar con 0s si no es múltiplo de 4
    while len(cadena) % 4 != 0:
        cadena += "0"
        print("Nota: Se agregó un '0' al final para completar el bloque de 4 bits.")

    hamming = HammingChannel()
    
    print("\nResultados paso a paso:")
    print(f"{'Bloque Original':<15} | {'Codificado (7 bits)':<20} | {'Con Ruido':<20} | {'Decodificado':<15} | {'Estado'}")
    print("-" * 90)

    # Procesar en bloques de 4
    for i in range(0, len(cadena), 4):
        chunk_str = cadena[i:i+4]
        nibble = [int(b) for b in chunk_str]
        
        # 1. Codificar
        encoded = hamming.encode_block(nibble)
        
        # 2. Ruido (Forzamos ruido alto para ver que funcione)
        noisy = hamming.simulate_noise(encoded, error_prob=0.5) 
        
        # 3. Decodificar
        decoded, error_found = hamming.decode_block(noisy)
        
        # Formato para imprimir
        enc_str = ''.join(map(str, encoded))
        noisy_str = ''.join(map(str, noisy))
        dec_str = ''.join(map(str, decoded))
        status = "CORREGIDO" if error_found else "OK"
        
        print(f"{chunk_str:<15} | {enc_str:<20} | {noisy_str:<20} | {dec_str:<15} | {status}")

# --- MENÚ PRINCIPAL ---
if __name__ == "__main__":
    while True:
        print("\n" + "="*40)
        print(" PROYECTO FINAL REDES - CÓDIGO HAMMING")
        print("="*40)
        print("1. Simular transmisión de IMAGEN (luna.jpg)")
        print("2. Simular cadena de BITS manual")
        print("3. Salir")
        
        opcion = input("\nSelecciona una opción (1-3): ")

        if opcion == "1":
            nombre_archivo = "luna.jpg" # <--- TU IMAGEN AQUÍ
            procesar_imagen(nombre_archivo, noise_level=0.15)
        elif opcion == "2":
            procesar_cadena_manual()
        elif opcion == "3":
            print("Saliendo del sistema...")
            break
        else:
            print("Opción no válida.")