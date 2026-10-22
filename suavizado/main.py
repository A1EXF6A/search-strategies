import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

from core import (
    apply_frequency_filter,
    filter_image,
    frequency_filter_image,
    imnoise,
    kernel_sum,
    parse_kernel,
)

SIZES = (3, 5, 7)
FILTERS = [
    ("mask", "Mascara personalizada"),
    ("mean", "Media"),
    ("median", "Mediana"),
    ("mode", "Moda"),
]
FREQ_FILTERS = [
    ("ideal", "Ideal"),
    ("butterworth", "Butterworth"),
    ("gaussian", "Gaussiano"),
]
BOX = 420
IMAGE_TYPES = [
    ("Imagenes", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff"),
    ("Todos los archivos", "*.*"),
]


class App:
    def __init__(self, root):
        self.root = root
        root.title("Suavizado por mascara")
        self.clean = None
        self.original = None
        self.result = None
        self.path = None
        self.running = False
        self.pending = None
        self.kernel_size = tk.IntVar(value=3)
        self.use_padding = tk.BooleanVar(value=True)
        self.noise_density = tk.StringVar(value="0.05")
        self.filter_kind = tk.StringVar(value="mask")
        self.step = tk.StringVar(value="1")
        self.mode = tk.StringVar(value="espacial")
        self.freq_pass = tk.StringVar(value="low")
        self.freq_shape = tk.StringVar(value="gaussian")
        self.freq_cutoff = tk.StringVar(value="30")
        self.freq_order = tk.StringVar(value="2")
        self._build()
        self._build_kernel_grid(3)

    def _build(self):
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=6)
        tk.Button(top, text="Cargar imagen...", command=self.load_image).pack(
            side="left"
        )
        tk.Label(top, text="Ruido sal y pimienta (densidad):").pack(
            side="left", padx=(10, 2)
        )
        tk.Entry(top, textvariable=self.noise_density, width=6).pack(side="left")
        self.btn_noise = tk.Button(
            top, text="Aplicar ruido", command=self.apply_noise, state="disabled"
        )
        self.btn_noise.pack(side="left", padx=(6, 0))
        self.status = tk.Label(top, text="Ninguna imagen cargada", anchor="w")
        self.status.pack(side="left", padx=10)

        images = tk.Frame(self.root)
        images.pack(fill="both", expand=True, padx=8)
        self.lbl_original = self._image_panel(images, "Imagen original")
        self.lbl_result = self._image_panel(images, "Imagen procesada")

        controls = tk.LabelFrame(self.root, text="Filtro")
        controls.pack(fill="x", padx=8, pady=6)

        self.btn_mode = tk.Button(
            controls, text="Modo: dominio espacial", command=self.toggle_mode
        )

        kinds = tk.Frame(controls)
        tk.Label(kinds, text="Tipo:").pack(side="left")
        for value, text in FILTERS:
            tk.Radiobutton(
                kinds,
                text=text,
                variable=self.filter_kind,
                value=value,
                command=self._filter_changed,
            ).pack(side="left")

        sizes = tk.Frame(controls)
        tk.Label(sizes, text="Tamano:").pack(side="left")
        for size in SIZES:
            tk.Radiobutton(
                sizes,
                text="%dx%d" % (size, size),
                variable=self.kernel_size,
                value=size,
                command=lambda: self._build_kernel_grid(self.kernel_size.get()),
            ).pack(side="left")
        tk.Label(sizes, text="Salto:").pack(side="left", padx=(12, 2))
        tk.Entry(sizes, textvariable=self.step, width=4).pack(side="left")

        self._build_freq_frame(controls)

        self.kernel_frame = tk.Frame(controls)

        self.padding_check = tk.Checkbutton(
            controls,
            text="Usar padding (borde negro; con salto 1 la salida es del tamano de la imagen)",
            variable=self.use_padding,
        )

        actions = tk.Frame(controls)
        self.btn_process = tk.Button(actions, text="Procesar", command=self.process)
        self.btn_process.pack(side="left")
        self.btn_save = tk.Button(
            actions,
            text="Guardar resultado...",
            command=self.save_result,
            state="disabled",
        )
        self.btn_save.pack(side="left", padx=6)
        self.progress = tk.Label(actions, text="", anchor="w")
        self.progress.pack(side="left", padx=10)

        # Orden de pintado dentro del panel y que controles pertenecen a
        # cada modo: "spatial" solo dominio espacial, "freq" solo FFT,
        # "both" en los dos.
        self._controls = [
            (self.btn_mode, {"anchor": "w"}, "both"),
            (kinds, {"anchor": "w"}, "spatial"),
            (sizes, {"anchor": "w"}, "spatial"),
            (self.freq_frame, {"anchor": "w", "pady": 4}, "freq"),
            (self.kernel_frame, {"anchor": "w", "pady": 4}, "spatial"),
            (self.padding_check, {"anchor": "w"}, "spatial"),
            (actions, {"fill": "x", "pady": 4}, "both"),
        ]
        self._apply_mode()

    def _build_freq_frame(self, parent):
        """Controles del filtrado en el dominio de la frecuencias."""
        self.freq_frame = tk.Frame(parent)

        passes = tk.Frame(self.freq_frame)
        passes.pack(anchor="w")
        tk.Label(passes, text="Paso:").pack(side="left")
        for value, text in (("low", "Paso bajo"), ("high", "Paso alto")):
            tk.Radiobutton(
                passes, text=text, variable=self.freq_pass, value=value
            ).pack(side="left")

        shapes = tk.Frame(self.freq_frame)
        shapes.pack(anchor="w")
        tk.Label(shapes, text="Forma:").pack(side="left")
        for value, text in FREQ_FILTERS:
            tk.Radiobutton(
                shapes, text=text, variable=self.freq_shape, value=value
            ).pack(side="left")

        params = tk.Frame(self.freq_frame)
        params.pack(anchor="w")
        tk.Label(params, text="Corte D0 (ciclos/imagen):").pack(side="left")
        tk.Entry(params, textvariable=self.freq_cutoff, width=6).pack(
            side="left", padx=(2, 8)
        )
        tk.Label(params, text="Orden n (Butterworth):").pack(side="left")
        tk.Entry(params, textvariable=self.freq_order, width=4).pack(
            side="left", padx=2
        )
        tk.Button(
            params, text="Ver matriz del filtro...", command=self.show_filter_matrix
        ).pack(side="left", padx=8)

        tk.Label(
            self.freq_frame,
            text="FFT con numpy sobre la imagen cargada; la salida tiene su mismo tamano.",
            fg="#555555",
        ).pack(anchor="w", pady=(2, 0))

    def toggle_mode(self):
        """Cambia entre el dominio espacial y el dominio de la frecuencias."""
        if self.mode.get() == "espacial":
            self.mode.set("frecuencias")
            self.btn_mode.configure(text="Modo: dominio frecuencias")
        else:
            self.mode.set("espacial")
            self.btn_mode.configure(text="Modo: dominio espacial")
        self._apply_mode()

    def _apply_mode(self):
        """Muestra los controles del modo activo y esconde los demas."""
        espacial = self.mode.get() == "espacial"
        for widget, options, scope in self._controls:
            widget.pack_forget()
        for widget, options, scope in self._controls:
            if scope == "both" or (scope == "spatial") == espacial:
                widget.pack(**options)

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
        self._filter_changed()
        if self.filter_kind.get() == "mask":
            for widget in self.kernel_frame.grid_slaves(row=0, column=0):
                widget.focus_set()
        self._filter_changed()

    def _filter_changed(self):
        """La cuadricula solo se usa con la mascara personalizada."""
        state = "normal" if self.filter_kind.get() == "mask" else "disabled"
        for widget in self.kernel_frame.grid_slaves():
            widget.configure(state=state)

    def _read_cutoff(self):
        text = str(self.freq_cutoff.get()).strip().replace(",", ".")
        try:
            cutoff = float(text)
        except ValueError:
            cutoff = 0.0
        if cutoff <= 0:
            raise ValueError(
                "El corte D0 debe ser un numero mayor que 0 (ejemplo: 30)."
            )
        return cutoff

    def _read_order(self):
        try:
            order = int(str(self.freq_order.get()).strip())
        except ValueError:
            order = 0
        if order < 1:
            raise ValueError(
                "El orden n debe ser un entero mayor o igual a 1 (ejemplo: 2)."
            )
        return order

    def _freq_params(self):
        """Lee los controles de frecuencias.

        Devuelve (forma, corte, paso_alto, orden, resumen) o lanza
        ValueError con el mensaje para el usuario. El orden solo se pide
        para Butterworth; en el resto se ignora.
        """
        shape = self.freq_shape.get()
        cutoff = self._read_cutoff()
        order = self._read_order() if shape == "butterworth" else 1
        highpass = self.freq_pass.get() == "high"
        summary = "%s %s, D0=%g" % (
            "paso alto" if highpass else "paso bajo",
            dict(FREQ_FILTERS)[shape].lower(),
            cutoff,
        )
        if shape == "butterworth":
            summary += ", orden %d" % order
        return shape, cutoff, highpass, order, summary

    def show_filter_matrix(self):
        """Abre una ventana con la matriz H(u,v) centrada (negro=0, blanco=1)."""
        if self.original is None:
            messagebox.showinfo("Imagen", "Primero carga una imagen.")
            return
        try:
            shape, cutoff, highpass, order, summary = self._freq_params()
        except ValueError as exc:
            messagebox.showerror("Frecuencias invalidas", str(exc))
            return
        image = frequency_filter_image(
            shape, cutoff, highpass, order, self.original.size
        )
        if max(image.size) < BOX:
            factor = max(1, min(8, BOX // max(image.size)))
            image = image.resize(
                (image.width * factor, image.height * factor),
                Image.Resampling.NEAREST,
            )
        else:
            image.thumbnail((BOX, BOX), Image.Resampling.NEAREST)

        window = tk.Toplevel(self.root)
        window.title("Matriz del filtro H(u,v)")
        tk.Label(
            window,
            text="H(u,v) centrada (negro=0, blanco=1) | %s | imagen %dx%d"
            % (summary, self.original.size[0], self.original.size[1]),
        ).pack(padx=8, pady=(8, 4))
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(window, image=photo, bg="#dddddd")
        label.image = photo
        label.pack(padx=8, pady=(0, 8))

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Seleccionar imagen", filetypes=IMAGE_TYPES
        )
        if not path:
            return
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Error", "No se pudo abrir la imagen:\n%s" % exc)
            return
        self.clean = image
        self.path = path
        self.btn_noise.configure(state="normal")
        self._apply_noise()

    def apply_noise(self):
        """Vuelve a calcular el ruido con la densidad actual, desde la
        imagen limpia (no acumula ruido sobre la que ya esta en pantalla)."""
        if self.clean is None:
            messagebox.showinfo("Imagen", "Primero carga una imagen.")
            return
        self._apply_noise()

    def _apply_noise(self):
        try:
            density = float(self.noise_density.get().replace(",", "."))
            image = imnoise(self.clean, density=density)
        except ValueError:
            messagebox.showerror(
                "Densidad invalida",
                "Escribe un numero entre 0 y 1, por ejemplo 0.05.",
            )
            return
        self.original = image
        self.result = None
        self.btn_save.configure(state="disabled")
        self._show(self.lbl_original, image)
        self._clear(self.lbl_result)
        self.status.configure(
            text="%s  (%dx%d) | ruido sal y pimienta, densidad %g"
            % (self.path, image.size[0], image.size[1], density)
        )

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

        source = self.original
        megapixels = 0.0

        if self.mode.get() == "frecuencias":
            try:
                shape, cutoff, highpass, order, summary = self._freq_params()
            except ValueError as exc:
                messagebox.showerror("Frecuencias invalidas", str(exc))
                return
            summary = "FFT " + summary

            def compute():
                return apply_frequency_filter(source, shape, cutoff, highpass, order)
        else:
            kind = self.filter_kind.get()
            kernel = None
            if kind == "mask":
                try:
                    kernel = self._read_kernel()
                except ValueError as exc:
                    messagebox.showerror("Mascara invalida", str(exc))
                    return
                size = len(kernel)
            else:
                size = self.kernel_size.get()
            use_padding = self.use_padding.get()
            if not use_padding:
                width, height = source.size
                if width < size or height < size:
                    messagebox.showerror(
                        "Filtro invalido",
                        "Sin padding la imagen debe ser al menos del tamano del filtro.",
                    )
                    return

            try:
                step = int(str(self.step.get()).strip())
                if step < 1:
                    raise ValueError
            except ValueError:
                messagebox.showerror(
                    "Salto invalido",
                    "Escribe un entero mayor o igual a 1, por ejemplo 1.",
                )
                return

            if kind == "mask":
                summary = "suma de la mascara: %g" % kernel_sum(kernel)
            else:
                summary = "%s %dx%d" % (
                    dict(FILTERS)[kind].lower(),
                    size,
                    size,
                )
            if step > 1:
                summary += ", salto %d" % step

            def compute():
                return filter_image(
                    source,
                    kind,
                    kernel=kernel,
                    size=size,
                    use_padding=use_padding,
                    step=step,
                )

            # Solo el calculo en Python puro es lento; la FFT va con numpy.
            megapixels = source.size[0] * source.size[1] / 1_000_000.0

        if megapixels > 4 and not messagebox.askyesno(
            "Imagen grande",
            "La imagen tiene %.1f MP y el calculo es en Python puro:\n"
            "puede tardar bastante. Deseas continuar?" % megapixels,
        ):
            return

        self.running = True
        self._last_summary = summary
        self.btn_process.configure(state="disabled")
        self.progress.configure(text="Procesando...")
        self.root.update_idletasks()

        holder = {}

        def work():
            started = time.time()
            try:
                holder["image"] = compute()
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
        self.progress.configure(
            text="Listo en %.2fs | %s | salida: %dx%d"
            % (
                holder["elapsed"],
                self._last_summary,
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
