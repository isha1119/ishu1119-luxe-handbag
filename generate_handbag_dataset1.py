import pandas as pd
import random

# Step 1: Define possible values
subcategories = ["Backpack", "Clutch", "Satchel", "Shoulder Bag", "Sling Bag", "Tote"]
brands = ["LuxeCraft", "StyleNest"]
adjectives = ["Elegant", "Classic", "Modern", "Trendy", "Vintage", "Stylish"]
materials = ["Leather", "Canvas", "Suede", "Nylon", "Faux Leather", "Velvet"]

# Step 2: Generate 50 records
data = []
for i in range(1, 51):
    product_id = f"P{i:03d}"
    subcat = random.choice(subcategories)
    brand = random.choice(brands)
    product_name = f"{random.choice(adjectives)} {random.choice(materials)} {subcat}"
    price = round(random.uniform(900, 3000), 2)
    rating = round(random.uniform(1.5, 5.0), 1)

    data.append({
        "Product ID": product_id,
        "Product Name": product_name,
        "Subcategory": subcat,
        "Brand": brand,
        "Price (₹)": price,
        "Rating": rating
    })

# Step 3: Convert to DataFrame
df = pd.DataFrame(data)

# Step 4: Save to CSV
df.to_csv("handbags_two_brands_cleaned_00.csv", index=False)
print("✅ Dataset saved as handbags_two_brands_cleaned_00.csv")
