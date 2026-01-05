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
# CLASE DE ANIMACIÓN (VERSIÓN CORREGIDA Y RÁPIDA)
# =============================================================================
class AnimacionTablaHamming:
    def __init__(self, parent, bits_recibidos, hamming_instance, error_pos_real):
        self.top = tk.Toplevel(parent)
        self.top.title("Proceso de Decodificación Hamming (7,4)")
        self.top.geometry("750x450")
        self.top.configure(bg="white")
        
        # Aseguramos que siempre haya 7 bits (rellenando con ceros si falta alguno)
        self.bits = bits_recibidos
        while len(self.bits) < 7:
            self.bits.insert(0, 0)
            
        self.H = hamming_instance.H
        self.error_pos_real = error_pos_real
        self.labels = []
        self.pasos = []
        self.indice_paso = 0
        
        # --- GUI DE LA TABLA ---
        frame_tabla = tk.Frame(self.top, bg="white", padx=20, pady=20)
        frame_tabla.pack(expand=True, fill="both")
        
        headers = ["Etapa"] + [f"Bit {i}" for i in range(7)] + ["Resultado"]
        for j, h in enumerate(headers):
            tk.Label(frame_tabla, text=h, font=("Arial", 10, "bold"), 
                     borderwidth=1, relief="solid", width=8, bg="#ecf0f1").grid(row=0, column=j, sticky="nsew", ipady=5)

        self.row_names = ["Dato Recibido", "Prueba H-Fila 1", "Prueba H-Fila 2", "Prueba H-Fila 3", "Dato Corregido"]
        
        for i, nombre in enumerate(self.row_names):
            fila_labels = []
            tk.Label(frame_tabla, text=nombre, font=("Arial", 9, "bold"), 
                     borderwidth=1, relief="solid", width=15, anchor="w", bg="#ecf0f1").grid(row=i+1, column=0, sticky="nsew", padx=1)
            
            # Celdas bits
            for j in range(7):
                lbl = tk.Label(frame_tabla, text="", font=("Consolas", 12), borderwidth=1, relief="solid", bg="white")
                lbl.grid(row=i+1, column=j+1, sticky="nsew")
                fila_labels.append(lbl)
            
            # Celda resultado
            lbl_res = tk.Label(frame_tabla, text="", font=("Arial", 9, "bold"), borderwidth=1, relief="solid", bg="white", width=10)
            lbl_res.grid(row=i+1, column=8, sticky="nsew")
            fila_labels.append(lbl_res)
            
            self.labels.append(fila_labels)

        self.lbl_status = tk.Label(self.top, text="Iniciando...", font=("Arial", 11), bg="white", fg="blue")
        self.lbl_status.pack(pady=10)

        # Construir y ejecutar
        self.construir_pasos()
        # Tiempo inicial corto (200ms)
        self.top.after(200, self.ejecutar_siguiente_paso)

    def construir_pasos(self):
        # 1. Cargar bits (instantáneo visualmente)
        for i, bit in enumerate(self.bits):
            self.pasos.append(("set", 0, i, str(bit))) # Row 0 is Data
        self.pasos.append(("text", "Analizando bits recibidos..."))

        syndrome = []
        colores = ["#d4e6f1", "#d5f5e3", "#fcf3cf"]
        
        # 2. Comprobaciones de Paridad
        for row_idx in range(3): 
            h_row = self.H[row_idx]
            parity_sum = 0
            
            self.pasos.append(("text", f"Verificando Fila {row_idx+1} de Matriz H..."))
            
            for bit_idx, h_val in enumerate(h_row):
                if h_val == 1:
                    val_bit = self.bits[bit_idx]
                    parity_sum += val_bit
                    # Acción: Marcar celda
                    self.pasos.append(("set", row_idx+1, bit_idx, str(val_bit)))
                    self.pasos.append(("bg", row_idx+1, bit_idx, colores[row_idx]))
            
            res = parity_sum % 2
            syndrome.append(res)
            
            color_res = "green" if res == 0 else "red"
            txt_res = "OK (0)" if res == 0 else "MAL (1)"
            self.pasos.append(("set_res", row_idx+1, txt_res, color_res))

        # 3. Corrección
        if np.any(syndrome):
            self.pasos.append(("text", f"¡Error detectado! Síndrome: {syndrome}"))
            
            # Buscar columna
            col_encontrada = -1
            syndrome_tuple = tuple(syndrome)
            for i in range(7):
                if tuple(self.H[:, i]) == syndrome_tuple:
                    col_encontrada = i
                    break
            
            if col_encontrada != -1:
                # Marcar columna verticalmente
                for r in range(1, 4):
                     if self.H[r-1][col_encontrada] == 1:
                        self.pasos.append(("bg", r, col_encontrada, "#e74c3c")) # Rojo
                
                self.pasos.append(("text", f"Error en Bit {col_encontrada}. Corrigiendo..."))
                
                bits_corregidos = list(self.bits).copy()
                bits_corregidos[col_encontrada] = 1 - bits_corregidos[col_encontrada]
                
                for i, bit in enumerate(bits_corregidos):
                    bg = "#5dade2" if i == col_encontrada else "white"
                    self.pasos.append(("set", 4, i, str(bit)))
                    self.pasos.append(("bg", 4, i, bg))
                
                self.pasos.append(("set_res", 4, "REPARADO", "blue"))
        else:
            self.pasos.append(("text", "Transmisión Correcta. Sin errores."))
            for i, bit in enumerate(self.bits):
                self.pasos.append(("set", 4, i, str(bit)))
            self.pasos.append(("set_res", 4, "INTACTO", "green"))

    def ejecutar_siguiente_paso(self):
        try:
            if self.indice_paso >= len(self.pasos):
                self.lbl_status.config(text="Proceso finalizado.", fg="black")
                return

            tipo = self.pasos[self.indice_paso][0]
            data = self.pasos[self.indice_paso]
            
            # Mapeo de índices
            # data[1] es fila (0..4) -> self.labels[0..4]
            # data[2] es columna bit (0..6) -> self.labels[row][0..6]
            
            if tipo == "set":
                self.labels[data[1]][data[2]].config(text=data[3])
            elif tipo == "bg":
                self.labels[data[1]][data[2]].config(bg=data[3])
            elif tipo == "set_res":
                # La celda de resultado es la última (-1)
                self.labels[data[1]][-1].config(text=data[2], fg=data[3])
            elif tipo == "text":
                self.lbl_status.config(text=data[1])

            self.indice_paso += 1
            # VELOCIDAD: 60ms (Bastante rápido para que no aburra)
            self.top.after(60, self.ejecutar_siguiente_paso)
            
        except Exception as e:
            print(f"Error en animación: {e}")


# =============================================================================
# 3. INTERFAZ GRÁFICA PRINCIPAL (GUI)
# =============================================================================
class HammingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Proyecto Final - Corrección Hamming (7,4)")
        self.root.geometry("950x700") 
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
    # PESTAÑA 1: IMAGEN (Igual que antes)
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
    # PESTAÑA 2: TEXTO (CON BOTÓN DE ANIMACIÓN ESTILO TABLA)
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

        ttk.Label(frame, text="Selecciona una fila para ver el proceso:", style="Header.TLabel").pack(anchor="w")

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

        # --- BOTÓN PARA ABRIR LA ANIMACIÓN DE TABLA ---
        btn_animar = tk.Button(frame, text="🎥 VER DETALLE TABLA HAMMING", 
                               bg="#e67e22", fg="white", font=("Segoe UI", 12, "bold"),
                               command=self.abrir_animacion_tabla)
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

    def abrir_animacion_tabla(self):
        seleccion = self.tree.selection()
        if not seleccion:
            messagebox.showwarning("Atención", "Primero selecciona una fila de la tabla.")
            return

        item = self.tree.item(seleccion[0])
        values = item['values']
        tags = item['tags']

        # --- CORRECCIÓN CLAVE ---
        # 1. Convertimos a string
        val_str = str(values[2])
        # 2. Rellenamos con ceros a la izquierda si faltan (ej: "101" -> "0000101")
        val_str = val_str.zfill(7)
        # 3. Convertimos a lista de enteros
        cadena_recibida = [int(x) for x in val_str]
        
        tag_info = tags[0] if tags else "ok"
        error_pos_real = -1
        
        if "error" in str(tag_info):
             try:
                error_pos_real = int(str(tag_info).split("_")[1])
             except:
                pass

        AnimacionTablaHamming(self.root, cadena_recibida, self.hamming, error_pos_real)


# =============================================================================
# 3. LANZAMIENTO
# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = HammingApp(root)
    root.mainloop()