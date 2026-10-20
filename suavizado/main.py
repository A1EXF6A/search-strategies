"""Interfaz basica para aplicar una mascara de suavizado a una imagen."""

import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

from core import apply_mask, kernel_sum, parse_kernel

SIZES = (3, 5, 7)
BOX = 420
IMAGE_TYPES = [
    ("Imagenes", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff"),
    ("Todos los archivos", "*.*"),
]


class App:
    def __init__(self, root):
        self.root = root
        root.title("Suavizado por mascara")
        self.original = None
        self.result = None
        self.running = False
        self.pending = None
        self.kernel_size = tk.IntVar(value=3)
        self.use_padding = tk.BooleanVar(value=True)
        self._build()
        self._build_kernel_grid(3)

    def _build(self):
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=6)
        tk.Button(top, text="Cargar imagen...", command=self.load_image).pack(side="left")
        self.status = tk.Label(top, text="Ninguna imagen cargada", anchor="w")
        self.status.pack(side="left", padx=10)

        images = tk.Frame(self.root)
        images.pack(fill="both", expand=True, padx=8)
        self.lbl_original = self._image_panel(images, "Imagen original")
        self.lbl_result = self._image_panel(images, "Imagen procesada")

        controls = tk.LabelFrame(self.root, text="Mascara")
        controls.pack(fill="x", padx=8, pady=6)

        sizes = tk.Frame(controls)
        sizes.pack(anchor="w")
        tk.Label(sizes, text="Tamano:").pack(side="left")
        for size in SIZES:
            tk.Radiobutton(
                sizes,
                text="%dx%d" % (size, size),
                variable=self.kernel_size,
                value=size,
                command=lambda: self._build_kernel_grid(self.kernel_size.get()),
            ).pack(side="left")

        self.kernel_frame = tk.Frame(controls)
        self.kernel_frame.pack(anchor="w", pady=4)

        tk.Checkbutton(
            controls,
            text="Usar padding (borde negro, salida del mismo tamano que la imagen)",
            variable=self.use_padding,
        ).pack(anchor="w")

        actions = tk.Frame(controls)
        actions.pack(fill="x", pady=4)
        self.btn_process = tk.Button(actions, text="Procesar", command=self.process)
        self.btn_process.pack(side="left")
        self.btn_save = tk.Button(
            actions, text="Guardar resultado...", command=self.save_result, state="disabled"
        )
        self.btn_save.pack(side="left", padx=6)
        self.progress = tk.Label(actions, text="", anchor="w")
        self.progress.pack(side="left", padx=10)

    def _image_panel(self, parent, title):
        frame = tk.LabelFrame(parent, text=title)
        frame.pack(side="left", expand=True, fill="both", padx=4, pady=4)
        label = tk.Label(frame, text="(sin imagen)", bg="#dddddd", width=50, height=20)
        label.pack(padx=6, pady=6, fill="both", expand=True)
        return label

    def _build_kernel_grid(self, size):
        for child in self.kernel_frame.winfo_children():
            child.destroy()
        self.cells = []
        for y in range(size):
            row = []
            for x in range(size):
                var = tk.StringVar()
                entry = tk.Entry(
                    self.kernel_frame, width=6, textvariable=var, justify="center"
                )
                entry.grid(row=y, column=x, padx=2, pady=2)
                row.append(var)
            self.cells.append(row)
        self.cells[0][0].set("1")
        for widget in self.kernel_frame.grid_slaves(row=0, column=0):
            widget.focus_set()

    def load_image(self):
        path = filedialog.askopenfilename(title="Seleccionar imagen", filetypes=IMAGE_TYPES)
        if not path:
            return
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Error", "No se pudo abrir la imagen:\n%s" % exc)
            return
        self.original = image
        self.result = None
        self.btn_save.configure(state="disabled")
        self._show(self.lbl_original, image)
        self._clear(self.lbl_result)
        self.status.configure(text="%s  (%dx%d)" % (path, image.size[0], image.size[1]))

    def _show(self, label, image):
        thumb = image.copy()
        thumb.thumbnail((BOX, BOX))
        photo = ImageTk.PhotoImage(thumb)
        label.configure(image=photo, text="")
        label.image = photo

    def _clear(self, label):
        label.configure(image="", text="(sin imagen)")
        label.image = None

    def _read_kernel(self):
        return parse_kernel([[var.get() for var in row] for row in self.cells])

    def process(self):
        if self.running:
            return
        if self.original is None:
            messagebox.showinfo("Imagen", "Primero carga una imagen.")
            return
        try:
            kernel = self._read_kernel()
        except ValueError as exc:
            messagebox.showerror("Mascara invalida", str(exc))
            return
        use_padding = self.use_padding.get()
        if not use_padding:
            width, height = self.original.size
            if width < len(kernel[0]) or height < len(kernel):
                messagebox.showerror(
                    "Mascara invalida",
                    "Sin padding la imagen debe ser al menos del tamano de la mascara.",
                )
                return

        megapixels = self.original.size[0] * self.original.size[1] / 1_000_000.0
        if megapixels > 4 and not messagebox.askyesno(
            "Imagen grande",
            "La imagen tiene %.1f MP y el calculo es en Python puro:\n"
            "puede tardar bastante. Deseas continuar?" % megapixels,
        ):
            return

        self.running = True
        self._last_kernel = kernel
        self.btn_process.configure(state="disabled")
        self.progress.configure(text="Procesando...")
        self.root.update_idletasks()

        source = self.original
        holder = {}

        def work():
            started = time.time()
            try:
                holder["image"] = apply_mask(source, kernel, use_padding)
                holder["error"] = None
            except Exception as exc:
                holder["error"] = exc
            holder["elapsed"] = time.time() - started

        threading.Thread(target=work, daemon=True).start()
        self.pending = holder
        self.root.after(50, self._poll)

    def _poll(self):
        if "elapsed" not in self.pending:
            self.root.after(50, self._poll)
            return
        holder = self.pending
        self.pending = None
        self.running = False
        self.btn_process.configure(state="normal")
        if holder["error"] is not None:
            self.progress.configure(text="")
            messagebox.showerror("Error", str(holder["error"]))
            return
        self.result = holder["image"]
        self._show(self.lbl_result, self.result)
        self.btn_save.configure(state="normal")
        total = kernel_sum(self._last_kernel)
        self.progress.configure(
            text="Listo en %.2fs | suma de la mascara: %g | salida: %dx%d"
            % (
                holder["elapsed"],
                total,
                self.result.size[0],
                self.result.size[1],
            )
        )

    def save_result(self):
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(
            title="Guardar resultado",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("BMP", "*.bmp"), ("JPEG", "*.jpg")],
        )
        if not path:
            return
        try:
            self.result.save(path)
        except Exception as exc:
            messagebox.showerror("Error", "No se pudo guardar:\n%s" % exc)
            return
        self.progress.configure(text="Guardado en %s" % path)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
