
"""
Write tables to an Excel workbook (.xlsx), one sheet per table:
bold header row, frozen header, filter buttons, readable column widths.
"""

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

HEADER_FONT = Font(bold=True, color="FFFFFF")
HEADER_FILL = PatternFill("solid", fgColor="305496")


def write_workbook(path, sheets):
    """sheets: [(title, header, rows)]; header None = text-only sheet,
    rows are then lines of text."""
    book = Workbook()
    book.remove(book.active)
    for title, header, rows in sheets:
        sheet = book.create_sheet(title[:31])
        if header is None:
            for line in rows:
                sheet.append([line])
            sheet.column_dimensions["A"].width = 110
            continue
        sheet.append(header)
        for cell in sheet[1]:
            cell.font = HEADER_FONT
            cell.fill = HEADER_FILL
            cell.alignment = Alignment(horizontal="center")
        for row in rows:
            sheet.append(list(row))
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for column, name in enumerate(header, start=1):
            longest = max([len(str(name))] + [
                len(str(row[column - 1])) for row in rows[:300]
                if column - 1 < len(row)])
            sheet.column_dimensions[get_column_letter(column)].width = \
                min(max(8, longest + 2), 45)
    book.save(path)
    return path
