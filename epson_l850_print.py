#!/usr/bin/env python3
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


PRINTERS = {
    "Auto Epson L850": "L850_ESCPR",
    "L850 Color": "L850_Color",
    "L850 Mono": "L850_Mono",
}

QUALITY_OPTIONS = {
    "Normal plain paper": "PLAIN_NORMAL",
    "High plain paper": "PLAIN_HIGH",
    "Photo paper normal": "PMPHOTO_NORMAL",
    "Photo paper high": "PMPHOTO_HIGH",
    "Matte paper high": "PMMATT_HIGH",
    "CD/DVD high": "CDDVD_HIGH",
}

PAGE_SIZES = ["A4", "A5", "A6", "Letter", "Legal", "4X6FULL", "8x10"]

PAGE_SETS = {
    "All pages": "",
    "Odd pages only": "odd",
    "Even pages only": "even",
}


class EpsonPrintApp(tk.Tk):
    def __init__(self, initial_file: str = "") -> None:
        super().__init__()
        self.title("Epson L850 Print")
        self.geometry("560x460")
        self.minsize(520, 420)

        self.file_path = tk.StringVar(value=initial_file)
        self.printer = tk.StringVar(value="Auto Epson L850")
        self.color_mode = tk.StringVar(value="COLOR")
        self.quality = tk.StringVar(value="Normal plain paper")
        self.page_size = tk.StringVar(value="A4")
        self.page_set = tk.StringVar(value="All pages")
        self.copies = tk.IntVar(value=1)
        self.page_range = tk.StringVar()

        self._build()

    def _build(self) -> None:
        root = ttk.Frame(self, padding=16)
        root.pack(fill="both", expand=True)
        root.columnconfigure(1, weight=1)

        ttk.Label(root, text="Subor").grid(row=0, column=0, sticky="w", pady=6)
        file_row = ttk.Frame(root)
        file_row.grid(row=0, column=1, sticky="ew", pady=6)
        file_row.columnconfigure(0, weight=1)
        ttk.Entry(file_row, textvariable=self.file_path).grid(row=0, column=0, sticky="ew")
        ttk.Button(file_row, text="Vybrat", command=self.choose_file).grid(row=0, column=1, padx=(8, 0))

        ttk.Label(root, text="Tlaciaren").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Combobox(
            root,
            textvariable=self.printer,
            values=list(PRINTERS.keys()),
            state="readonly",
        ).grid(row=1, column=1, sticky="ew", pady=6)

        ttk.Label(root, text="Farba").grid(row=2, column=0, sticky="w", pady=6)
        color_row = ttk.Frame(root)
        color_row.grid(row=2, column=1, sticky="w", pady=6)
        ttk.Radiobutton(color_row, text="Farebne", value="COLOR", variable=self.color_mode).pack(side="left")
        ttk.Radiobutton(color_row, text="Ciernobielo", value="MONO", variable=self.color_mode).pack(
            side="left", padx=(16, 0)
        )

        ttk.Label(root, text="Kvalita").grid(row=3, column=0, sticky="w", pady=6)
        ttk.Combobox(
            root,
            textvariable=self.quality,
            values=list(QUALITY_OPTIONS.keys()),
            state="readonly",
        ).grid(row=3, column=1, sticky="ew", pady=6)

        ttk.Label(root, text="Papier").grid(row=4, column=0, sticky="w", pady=6)
        ttk.Combobox(root, textvariable=self.page_size, values=PAGE_SIZES, state="readonly").grid(
            row=4, column=1, sticky="ew", pady=6
        )

        ttk.Label(root, text="Strany").grid(row=5, column=0, sticky="w", pady=6)
        pages_row = ttk.Frame(root)
        pages_row.grid(row=5, column=1, sticky="ew", pady=6)
        pages_row.columnconfigure(1, weight=1)
        ttk.Label(pages_row, text="Rozsah").grid(row=0, column=0, sticky="w")
        ttk.Entry(pages_row, textvariable=self.page_range).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        ttk.Label(root, text="Duplex").grid(row=6, column=0, sticky="w", pady=6)
        ttk.Combobox(
            root,
            textvariable=self.page_set,
            values=list(PAGE_SETS.keys()),
            state="readonly",
        ).grid(row=6, column=1, sticky="ew", pady=6)

        ttk.Label(root, text="Kopie").grid(row=7, column=0, sticky="w", pady=6)
        ttk.Spinbox(root, from_=1, to=99, textvariable=self.copies, width=8).grid(
            row=7, column=1, sticky="w", pady=6
        )

        hint = (
            "Manualny duplex: najprv vytlac Odd pages only, potom otoc papier "
            "a vytlac Even pages only."
        )
        ttk.Label(root, text=hint, wraplength=500).grid(row=8, column=0, columnspan=2, sticky="ew", pady=(14, 4))

        buttons = ttk.Frame(root)
        buttons.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(18, 0))
        buttons.columnconfigure(0, weight=1)
        ttk.Button(buttons, text="Tlacit", command=self.print_file).grid(row=0, column=1, sticky="e")
        ttk.Button(buttons, text="Test CUPS", command=self.test_printers).grid(row=0, column=0, sticky="w")

    def choose_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Vyber subor na tlac",
            filetypes=[
                ("Dokumenty", "*.pdf *.txt *.png *.jpg *.jpeg *.odt *.doc *.docx"),
                ("Vsetky subory", "*"),
            ],
        )
        if path:
            self.file_path.set(path)

    def selected_printer(self) -> str:
        if self.printer.get() == "L850 Color":
            return PRINTERS["L850 Color"]
        if self.printer.get() == "L850 Mono":
            return PRINTERS["L850 Mono"]
        return PRINTERS["Auto Epson L850"]

    def build_command(self) -> list[str]:
        path = Path(self.file_path.get()).expanduser()
        if not path.exists():
            raise ValueError("Vyber existujuci subor.")

        mode = self.color_mode.get()
        color_mode = "monochrome" if mode == "MONO" else "color"
        command = [
            "lp",
            "-d",
            self.selected_printer(),
            "-n",
            str(max(1, int(self.copies.get()))),
            "-o",
            f"Ink={mode}",
            "-o",
            f"print-color-mode={color_mode}",
            "-o",
            f"MediaType={QUALITY_OPTIONS[self.quality.get()]}",
            "-o",
            f"PageSize={self.page_size.get()}",
        ]

        page_set = PAGE_SETS[self.page_set.get()]
        if page_set:
            command.extend(["-o", f"page-set={page_set}"])

        pages = self.page_range.get().strip()
        if pages:
            command.extend(["-P", pages])

        command.append(str(path))
        return command

    def print_file(self) -> None:
        try:
            command = self.build_command()
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        except Exception as exc:
            messagebox.showerror("Tlac zlyhala", str(exc))
            return

        output = result.stdout.strip() or "Uloha bola odoslana do tlaciarne."
        messagebox.showinfo("Odoslane", output)

    def test_printers(self) -> None:
        try:
            result = subprocess.run(["lpstat", "-p"], check=True, capture_output=True, text=True)
        except Exception as exc:
            messagebox.showerror("CUPS chyba", str(exc))
            return
        messagebox.showinfo("CUPS tlaciarne", result.stdout.strip())


if __name__ == "__main__":
    initial = sys.argv[1] if len(sys.argv) > 1 else ""
    app = EpsonPrintApp(initial)
    app.mainloop()
