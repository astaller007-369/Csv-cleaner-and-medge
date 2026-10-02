# CSV cleaner, filler & builder
Upload two CSVs, make numbers whole (xG columns untouched), copy data into blank cells only (matched on date + home + away),
build a new CSV in your click order, add percentage columns, delete columns, download.

Run: `pip install -r requirements.txt && streamlit run app.py` (or deploy `app.py` on Streamlit Community Cloud).
Downloads are UTF-8 with BOM so Excel shows accents correctly.
