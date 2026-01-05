import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# =============================================================================
# 1. MOTOR LÓGICO (CEREBRO MATEMÁTICO)
# =============================================================================
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
        
        error_detected = False
        corrected_vector = r_vector.copy()
        error_pos = -1 # -1 significa sin error

        if np.any(syndrome):
            error_detected = True
            syndrome_tuple = tuple(syndrome)
            if syndrome_tuple in self.syndrome_map:
                error_pos = self.syndrome_map[syndrome_tuple]
                corrected_vector[error_pos] = 1 - corrected_vector[error_pos]
        
        decoded_data = corrected_vector[:4]
        return decoded_data, error_detected, syndrome, error_pos

    def simulate_noise(self, encoded_msg, error_prob):
        noisy_msg = encoded_msg.copy()
        if random.random() < error_prob:
            bit_to_flip = random.randint(0, 6)
            noisy_msg[bit_to_flip] = 1 - noisy_msg[bit_to_flip]
        return noisy_msg

# =============================================================================
# 2. INTERFAZ GRÁFICA (GUI)
# =============================================================================
class HammingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Proyecto Final - Corrección Hamming (7,4)")
        self.root.geometry("950x700") # Un poco más alto para el botón nuevo
        self.hamming = HammingChannel()
        self.selected_image_path = None

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"), foreground="#2c3e50")
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        style.configure("Treeview", rowheight=25)

        # --- TÍTULO ---
        title_frame = tk.Frame(root, bg="#2980b9", pady=15)
        title_frame.pack(fill="x")
        tk.Label(title_frame, text="SISTEMA DE TRANSMISIÓN Y CORRECCIÓN DE DATOS", 
                 font=("Segoe UI", 16, "bold"), bg="#2980b9", fg="white").pack()

        # --- PESTAÑAS ---
        tab_control = ttk.Notebook(root)
        self.tab_imagen = ttk.Frame(tab_control)
        self.tab_texto = ttk.Frame(tab_control)
        
        tab_control.add(self.tab_imagen, text='  Simulación IMAGEN  ')
        tab_control.add(self.tab_texto, text='  Simulación DATOS (Bits)  ')
        tab_control.pack(expand=1, fill="both", padx=10, pady=10)

        self.setup_tab_imagen()
        self.setup_tab_texto()

    # -------------------------------------------------------------------------
    # PESTAÑA 1: IMAGEN
    # -------------------------------------------------------------------------
    def setup_tab_imagen(self):
        frame = ttk.Frame(self.tab_imagen, padding=20)
        frame.pack(fill="both", expand=True)

        lbl_inst = ttk.Label(frame, text="1. Cargar Imagen:", style="Header.TLabel")
        lbl_inst.pack(anchor="w", pady=(0, 5))
        btn_browse = ttk.Button(frame, text="📂 Seleccionar Archivo", command=self.select_file)
        btn_browse.pack(anchor="w", pady=5)
        self.lbl_path = ttk.Label(frame, text="Sin archivo...", foreground="gray")
        self.lbl_path.pack(anchor="w", pady=5)

        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=15)

        lbl_noise = ttk.Label(frame, text="2. Nivel de Ruido (Interferencia):", style="Header.TLabel")
        lbl_noise.pack(anchor="w", pady=(0, 5))
        noise_frame = ttk.Frame(frame)
        noise_frame.pack(anchor="w", fill="x")
        ttk.Label(noise_frame, text="Probabilidad:").pack(side="left")
        self.noise_slider = ttk.Scale(noise_frame, from_=0, to=100, orient='horizontal', length=300)
        self.noise_slider.set(15) 
        self.noise_slider.pack(side="left", padx=10)
        self.lbl_noise_val = ttk.Label(noise_frame, text="15%")
        self.lbl_noise_val.pack(side="left")
        self.noise_slider.configure(command=lambda v: self.lbl_noise_val.configure(text=f"{int(float(v))}%"))

        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=15)

        btn_run = ttk.Button(frame, text="🚀 EJECUTAR SIMULACIÓN VISUAL", command=self.run_image_simulation)
        btn_run.pack(fill="x", pady=5)
        self.lbl_status = ttk.Label(frame, text="Listo.", foreground="gray")
        self.lbl_status.pack()

    def select_file(self):
        filename = filedialog.askopenfilename(title="Seleccionar Imagen", filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")])
        if filename:
            self.selected_image_path = filename
            self.lbl_path.config(text=f"Archivo: ...{filename[-30:]}", foreground="green")

    def run_image_simulation(self):
        if not self.selected_image_path:
            messagebox.showwarning("Atención", "Selecciona una imagen primero.")
            return
        noise_prob = self.noise_slider.get() / 100.0
        self.lbl_status.config(text="Procesando...", foreground="blue")
        self.root.update()

        try:
            img = Image.open(self.selected_image_path).convert('L')
            img = img.resize((150, 150)) 
            img_arr = np.array(img)
            original_shape = img_arr.shape
            flattened_pixels = img_arr.flatten()

            encoded_stream = []
            noisy_stream = []
            corrected_stream = []
            total_errors = 0

            for pixel in flattened_pixels:
                bits = self.hamming.text_to_bits(pixel, 8)
                n_high, n_low = bits[:4], bits[4:]
                
                parts_reconstructed = []
                for nibble in [n_high, n_low]:
                    encoded = self.hamming.encode_block(nibble)
                    noisy = self.hamming.simulate_noise(encoded, noise_prob)
                    decoded_nibble, error_found, _, _ = self.hamming.decode_block(noisy)
                    noisy_stream.append(noisy) 
                    if error_found: total_errors += 1
                    parts_reconstructed.extend(decoded_nibble)
                corrected_pixel = self.hamming.bits_to_text(parts_reconstructed)
                corrected_stream.append(corrected_pixel)

            noisy_pixels = []
            for i in range(0, len(noisy_stream), 2):
                n1 = noisy_stream[i][:4]
                n2 = noisy_stream[i+1][:4]
                val = self.hamming.bits_to_text(np.concatenate((n1, n2)))
                noisy_pixels.append(val)
            
            img_noisy = np.array(noisy_pixels).reshape(original_shape)
            img_corrected = np.array(corrected_stream).reshape(original_shape)
            self.lbl_status.config(text=f"Errores corregidos: {total_errors}", foreground="green")

            plt.figure(figsize=(12, 5))
            plt.suptitle(f"Análisis: {total_errors} errores corregidos exitosamente", fontsize=14)
            plt.subplot(1, 3, 1); plt.title("Original"); plt.imshow(img_arr, cmap='gray'); plt.axis('off')
            plt.subplot(1, 3, 2); plt.title("Señal con Ruido"); plt.imshow(img_noisy, cmap='gray'); plt.axis('off')
            plt.subplot(1, 3, 3); plt.title("Restaurada (Hamming)"); plt.imshow(img_corrected, cmap='gray'); plt.axis('off')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # -------------------------------------------------------------------------
    # PESTAÑA 2: TEXTO (CON BOTÓN DE ANIMACIÓN)
    # -------------------------------------------------------------------------
    def setup_tab_texto(self):
        frame = ttk.Frame(self.tab_texto, padding=20)
        frame.pack(fill="both", expand=True)

        input_frame = ttk.LabelFrame(frame, text=" Entrada de Datos ", padding=10)
        input_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(input_frame, text="Cadena Binaria:").pack(side="left")
        self.entry_bits = ttk.Entry(input_frame, width=30, font=("Consolas", 11))
        self.entry_bits.pack(side="left", padx=10)
        self.entry_bits.insert(0, "10110010") 
        
        btn_proc_bits = ttk.Button(input_frame, text="Procesar", command=self.run_text_simulation)
        btn_proc_bits.pack(side="left", padx=10)

        noise_frame = ttk.Frame(input_frame)
        noise_frame.pack(side="left", padx=20)
        ttk.Label(noise_frame, text="Ruido:").pack(side="left")
        self.txt_noise_slider = ttk.Scale(noise_frame, from_=0, to=100, orient='horizontal', length=100)
        self.txt_noise_slider.set(50) 
        self.txt_noise_slider.pack(side="left", padx=5)

        ttk.Label(frame, text="Resultados de la transmisión:", style="Header.TLabel").pack(anchor="w")

        columns = ("bloque", "codificado", "ruidoso", "estado")
        self.tree = ttk.Treeview(frame, columns=columns, show='headings', height=10)
        
        self.tree.heading("bloque", text="Bloque (4b)")
        self.tree.heading("codificado", text="Enviado (7b)")
        self.tree.heading("ruidoso", text="Recibido (7b)")
        self.tree.heading("estado", text="Estado del Paquete")
        
        self.tree.column("bloque", anchor="center")
        self.tree.column("codificado", anchor="center")
        self.tree.column("ruidoso", anchor="center")
        self.tree.column("estado", anchor="center")

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        self.tree.pack(side="top", fill="both", expand=True, pady=5)
        scrollbar.pack(side="right", fill="y")

        # --- AQUÍ ESTÁ EL BOTÓN NUEVO ---
        btn_animar = tk.Button(frame, text="🎥 VER ANIMACIÓN DE REPARACIÓN", 
                               bg="#e67e22", fg="white", font=("Segoe UI", 12, "bold"),
                               command=self.abrir_animacion_desde_boton)
        btn_animar.pack(fill="x", pady=10)

    def run_text_simulation(self):
        for i in self.tree.get_children(): self.tree.delete(i)
        cadena = self.entry_bits.get()
        noise_prob = self.txt_noise_slider.get() / 100.0

        if not cadena or not all(c in '01' for c in cadena):
            messagebox.showerror("Error", "Solo 0s y 1s.")
            return
        while len(cadena) % 4 != 0: cadena += "0"

        for i in range(0, len(cadena), 4):
            chunk_str = cadena[i:i+4]
            nibble = [int(b) for b in chunk_str]
            encoded = self.hamming.encode_block(nibble)
            noisy = self.hamming.simulate_noise(encoded, noise_prob)
            decoded, error_found, _, error_pos = self.hamming.decode_block(noisy)
            
            enc_str = ''.join(map(str, encoded))
            noisy_str = ''.join(map(str, noisy))
            
            tag_name = f"error_{error_pos}" if error_found else "ok"
            display_status = "⚠️ ERROR DETECTADO" if error_found else "✅ INTEGRO"
            
            self.tree.insert("", "end", values=(chunk_str, enc_str, noisy_str, display_status), tags=(tag_name,))

        self.tree.tag_configure("ok", foreground="green")
        for i in range(7):
            self.tree.tag_configure(f"error_{i}", foreground="red", background="#fadbd8")

    # -------------------------------------------------------------------------
    # ANIMACIÓN DE REPARACIÓN (TRIGGER DESDE BOTÓN)
    # -------------------------------------------------------------------------
    def abrir_animacion_desde_boton(self):
        # 1. Verificar selección
        seleccion = self.tree.selection()
        if not seleccion:
            messagebox.showwarning("Atención", "Primero selecciona una fila de la tabla (haz clic en ella).")
            return

        # 2. Obtener datos
        item = self.tree.item(seleccion[0])
        values = item['values']
        tags = item['tags']

        cadena_recibida = list(str(values[2]))
        tag_info = tags[0] if tags else "ok"
        
        # 3. Lanzar ventana
        top = tk.Toplevel(self.root)
        top.title("Escáner de Integridad de Datos")
        top.geometry("600x350")
        top.configure(bg="#2c3e50")

        lbl_titulo = tk.Label(top, text="ANALIZANDO PAQUETE...", font=("Arial", 16, "bold"), fg="white", bg="#2c3e50")
        lbl_titulo.pack(pady=20)

        frame_bits = tk.Frame(top, bg="#2c3e50")
        frame_bits.pack(pady=20)
        
        self.bit_labels = []
        for bit in cadena_recibida:
            l = tk.Label(frame_bits, text=bit, font=("Consolas", 24, "bold"), width=4, height=2, 
                         bg="white", fg="black", relief="raised", bd=3)
            l.pack(side="left", padx=5)
            self.bit_labels.append(l)

        lbl_log = tk.Label(top, text="Iniciando escaneo...", font=("Consolas", 12), fg="#f1c40f", bg="#2c3e50")
        lbl_log.pack(pady=20)

        # 4. Iniciar animación
        if "ok" in str(tag_info):
            self.animar_escaneo(top, 0, -1, lbl_log, lbl_titulo)
        else:
            try:
                error_pos = int(str(tag_info).split("_")[1])
                self.animar_escaneo(top, 0, error_pos, lbl_log, lbl_titulo)
            except:
                self.animar_escaneo(top, 0, -1, lbl_log, lbl_titulo)

    def animar_escaneo(self, window, index, error_pos, lbl_log, lbl_titulo):
        if index > 0:
            prev_idx = index - 1
            if prev_idx != error_pos:
                self.bit_labels[prev_idx].config(bg="#27ae60", fg="white")

        if index >= 7:
            if error_pos == -1:
                lbl_titulo.config(text="PAQUETE CORRECTO", fg="#27ae60")
                lbl_log.config(text="Escaneo finalizado. Sin errores.", fg="#27ae60")
            return

        lbl_log.config(text=f"Verificando bit en posición {index}...", fg="#f1c40f")
        self.bit_labels[index].config(bg="#f1c40f", fg="black")

        if index == error_pos:
            self.root.after(800, lambda: self.animar_error_encontrado(window, index, lbl_log, lbl_titulo))
        else:
            self.root.after(400, lambda: self.animar_escaneo(window, index + 1, error_pos, lbl_log, lbl_titulo))

    def animar_error_encontrado(self, window, index, lbl_log, lbl_titulo):
        self.bit_labels[index].config(bg="#e74c3c", fg="white")
        lbl_titulo.config(text="¡ERROR DETECTADO!", fg="#e74c3c")
        lbl_log.config(text=f"Corrupción de datos en bit {index}.", fg="#e74c3c")
        self.root.after(1500, lambda: self.animar_correccion(window, index, lbl_log, lbl_titulo))

    def animar_correccion(self, window, index, lbl_log, lbl_titulo):
        valor_actual = self.bit_labels[index].cget("text")
        nuevo_valor = "0" if valor_actual == "1" else "1"
        self.bit_labels[index].config(text=nuevo_valor, bg="#3498db", fg="white")
        lbl_titulo.config(text="REPARACIÓN EXITOSA", fg="#3498db")
        lbl_log.config(text=f"Bit {index} invertido y corregido.", fg="#3498db")

# =============================================================================
# 3. LANZAMIENTO
# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = HammingApp(root)
    root.mainloop()