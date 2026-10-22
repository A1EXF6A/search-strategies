"""GUI de deteccion de objetos.

Carga una imagen, umbraliza con un valor que da el usuario y muestra las
dos imagenes: la original y la copia con los rectangulos y centros de
cada objeto detectado con regionprops.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from core import detectar, dibujar

BOX = 420
IMAGE_TYPES = [
    ("Imagenes", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff"),
    ("Todos los archivos", "*.*"),
]


class App:
    def __init__(self, root):
        self.root = root
        root.title("Deteccion de objetos con regionprops")
        self.original = None
        self.resultado = None
        self.path = None
        self.umbral = tk.StringVar(value="128")
        self.area_min = tk.StringVar(value="1")
        self.invertir = tk.BooleanVar(value=False)
        self._build()

    def _build(self):
        acciones = tk.Frame(self.root)
        acciones.pack(fill="x", padx=8, pady=6)

        tk.Button(acciones, text="Cargar imagen...",
                  command=self.load_image).pack(side="left")
        tk.Label(acciones, text="Umbral (0-255):").pack(
            side="left", padx=(10, 2))
        self.entry_umbral = tk.Entry(acciones, textvariable=self.umbral,
                                     width=5)
        self.entry_umbral.pack(side="left")
        self.entry_umbral.bind("<Return>", lambda _e: self.process())
        tk.Checkbutton(acciones, text="Objetos claros",
                       variable=self.invertir,
                       command=self._auto).pack(side="left", padx=(6, 2))
        tk.Label(acciones, text="Area minima (px):").pack(
            side="left", padx=(6, 2))
        self.entry_area = tk.Entry(acciones, textvariable=self.area_min,
                                   width=6)
        self.entry_area.pack(side="left")
        self.entry_area.bind("<Return>", lambda _e: self.process())
        tk.Button(acciones, text="Detectar",
                  command=self.process).pack(side="left", padx=(8, 0))
        self.btn_save = tk.Button(acciones, text="Guardar resultado...",
                                  command=self.save_result, state="disabled")
        self.btn_save.pack(side="left", padx=(4, 0))
        self.status = tk.Label(acciones, text="Ninguna imagen cargada",
                               anchor="w")
        self.status.pack(side="left", padx=10)

        paneles = tk.Frame(self.root)
        paneles.pack(fill="both", expand=True, padx=8)
        self.lbl_original = self._panel(paneles, "Imagen original")
        self.lbl_result = self._panel(paneles, "Imagen con detecciones")

        marco = tk.LabelFrame(self.root, text="Objetos detectados")
        marco.pack(fill="both", padx=8, pady=6)

        columnas = ("n", "cx", "cy", "ancho", "alto", "area")
        self.tabla = ttk.Treeview(marco, columns=columnas, show="headings",
                                  height=6)
        for col, texto, ancho in (
            ("n", "#", 40), ("cx", "Centro X", 90), ("cy", "Centro Y", 90),
            ("ancho", "Ancho", 80), ("alto", "Alto", 80),
            ("area", "Area (px)", 100),
        ):
            self.tabla.heading(col, text=texto)
            self.tabla.column(col, width=ancho, anchor="center")
        scroll = ttk.Scrollbar(marco, orient="vertical",
                               command=self.tabla.yview)
        self.tabla.configure(yscrollcommand=scroll.set)
        self.tabla.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def _panel(self, parent, titulo):
        marco = tk.LabelFrame(parent, text=titulo)
        marco.pack(side="left", expand=True, fill="both", padx=4, pady=4)
        label = tk.Label(marco, text="(sin imagen)", bg="#dddddd",
                         width=50, height=20)
        label.pack(padx=6, pady=6, fill="both", expand=True)
        return label

    def load_image(self):
        path = filedialog.askopenfilename(title="Seleccionar imagen",
                                          filetypes=IMAGE_TYPES)
        if not path:
            return
        try:
            imagen = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Error",
                                 "No se pudo abrir la imagen:\n%s" % exc)
            return
        self.original = imagen
        self.path = path
        self.resultado = None
        self._show(self.lbl_original, imagen)
        self._clear_table()
        self.btn_save.configure(state="disabled")
        self.status.configure(text="%s  (%dx%d)" % (path, imagen.width,
                                                    imagen.height))
        self.process()

    def _leer_umbral(self):
        try:
            umbral = int(str(self.umbral.get()).strip())
        except ValueError:
            raise ValueError("El umbral debe ser un entero entre 0 y 255.")
        if not 0 <= umbral <= 255:
            raise ValueError("El umbral debe estar entre 0 y 255.")
        return umbral

    def _leer_area_min(self):
        try:
            area = int(str(self.area_min.get()).strip())
        except ValueError:
            raise ValueError("El area minima debe ser un entero >= 1.")
        return max(1, area)

    def _auto(self):
        if self.original is not None:
            self.process()

    def process(self):
        if self.original is None:
            messagebox.showinfo("Imagen", "Primero carga una imagen.")
            return
        try:
            umbral = self._leer_umbral()
            area_min = self._leer_area_min()
        except ValueError as exc:
            messagebox.showerror("Valor invalido", str(exc))
            return

        objetos = detectar(self.original, umbral,
                           invertir=self.invertir.get(), area_min=area_min)
        self.resultado = dibujar(self.original, objetos)
        self._show(self.lbl_result, self.resultado)
        self._fill_table(objetos)
        self.btn_save.configure(state="normal" if objetos else "disabled")
        self.status.configure(
            text="%s | umbral %d | %d objeto(s)" % (self.path, umbral,
                                                    len(objetos)))

    def _fill_table(self, objetos):
        self._clear_table()
        for i, obj in enumerate(objetos, start=1):
            cx, cy = obj["centro"]
            self.tabla.insert("", "end", values=(
                i, cx, cy, obj["ancho"], obj["alto"], obj["area"]))

    def _clear_table(self):
        for fila in self.tabla.get_children():
            self.tabla.delete(fila)

    def _show(self, label, imagen):
        thumb = imagen.copy()
        thumb.thumbnail((BOX, BOX))
        photo = ImageTk.PhotoImage(thumb)
        label.configure(image=photo, text="")
        label.image = photo

    def save_result(self):
        if self.resultado is None:
            return
        path = filedialog.asksaveasfilename(
            title="Guardar resultado", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("BMP", "*.bmp"),
                       ("JPEG", "*.jpg")])
        if not path:
            return
        try:
            self.resultado.save(path)
        except Exception as exc:
            messagebox.showerror("Error",
                                 "No se pudo guardar:\n%s" % exc)
            return
        self.status.configure(text="Guardado en %s" % path)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
