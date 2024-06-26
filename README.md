# Restaurant drive-thru orders

**Automates a food order from a json food menu using Retrieval-Augmented Generation (RAG) with OpenAI and MistralAI.**

These are the responses obtained from running the Python script using OpenAI

- "Hi, do you have cola?"

**MistralAI:** 'I'm sorry, based on the information provided, it appears that there are no cola options available in the drinks section ("Drinks") at this KFC location. You may consider other beverage options such as Iced Tea, Pepsi, 7Up, or Fanta. However, please note that all drinks listed are currently unavailable according to the context information.'

**OpenAI:** There is no specific mention of cola in the list provided. If you need further information or assistance, feel free to ask.
I will start by opening the file you uploaded to see its contents and check for information related to your question about cola. Let's begin by inspecting the file.

I have checked the file, and it contains information about the items available at KFC under different categories like Chicken, Drinks, and Burgers. Specifically, in the "Drinks" section, the following options are listed:

1. Iced Tea
2. Pepsi
3. 7Up
4. Fanta
5. Sourcy
6. Tropicana Apple
7. Guava
8. Tea
9. Latte
10. Espresso
11. Coffee
12. Sisi
13. Fernandes
14. Lipton


▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄

- "Hi I want to have a Fire Zinger Stacker without sauce and a cola"

**MistralAI:** 'Based on the context information provided, it seems that you are looking to order the "Fire Zinger Burger Meal" (M4) without sauce and a cola. According to the menu, the "Fire Zinger Burger Meal" comes with a "Fire Zinger Burger" (B9), a choice of side dish, and a drink in size M. To customize your order, you can specify that you would like to have the Fire Zinger Burger without sauce and a cola as your drink. Here is an example of how you might place your order:

"Hi, I would like to order the Fire Zinger Burger Meal (M4) without sauce on the burger and a cola as my drink. I would also like to choose [insert choice of side dish] as my side. Thank you."

I hope this helps! Let me know if you have any other questions.

Note: It is important to keep in mind that the context information provided may not be up-to-date or accurate, and it is always best to confirm the availability and details of menu items with the restaurant directly.'

**OpenAI:** I will open the file to check the available options and customize the order for you. Let's start by reviewing the file.

Based on the menu provided, you can choose the Fire Zinger Stacker from the Burgers section. Since you want it without sauce, I will order the Fire Zinger Stacker without sauce for you. Additionally, you requested a cola; I will check the available cola options from the Drinks section to include with your order. Let me find the cola options available.

The available cola options are Pepsi and 7Up. I will add a Pepsi to your order. Your order is a Fire Zinger Stacker without sauce and a Pepsi. I will proceed with placing this order for you. Thank you for your order!

▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄

- "Give me a Veggie Tender, medium, with salad"

**MistralAI:** Based on the provided context, there are several meal options that include Veggie Tenders (V2) and a medium-sized drink (L), but none explicitly mention a salad as a side dish. However, you could potentially customize one of the following options to include a salad:

* M20: 4 Veggie Tender meal - This meal includes 4 Veggie Tenders and a choice of one side dish. You could select a medium drink as your drink size and then ask to swap the included side dish for a salad.
* M37: Veggie Kids Meal - This meal includes 2 Veggie Tenders, a medium drink, a choice of one sauce, and a choice of one side dish. You could select a salad as your side dish.
* M39: Veggie Bucket - This meal includes 10 Veggie Tenders, a choice of two side dishes, and a medium drink. You could select a salad as one of your side dishes.

Please note that the availability of customizations may vary depending on the location and specific restaurant policies.

**OpenAI:** To proceed with your order request for a Veggie Tender, medium size, with salad, I will look for the availability and details of this item in the file you provided. Let's begin by checking the content of the file to see if this particular item is listed.

I have checked the menu, and I couldn't find a Veggie Tender with salad listed. However, there is a "Veggie Burger" available for 10 units. If you are interested in that alternative or want to explore other options, feel free to let me know.

▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄

- "Give me an orange chocolate milkshake, medium"

**MistralAI**: I'm sorry, there is no option for an orange chocolate milkshake, medium in the provided context information. The available drink options are not listed, only the sizes are mentioned for some menus. Please check the drink options and try again.

**OpenAI:** I couldn't find any information on an "Orange Chocolate Milkshake" in the document. It appears that the section I reviewed doesn't list this specific item, or it might not be available in the menu. If there's another item or drink you're interested in, feel free to ask

▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄

- "Give me the gluten free burger options "

**MistralAI:** The gluten-free burger options, denoted as "GF", are:

GF: "B2: Original Fillet Breaded", "B3: Original Fillet", "B9: Fire Zinger", "B10: Colonel Stacker", "B11: Colonel Burger", "B14: Original Burger"

Please note that this information is based on the context provided and may not reflect actual gluten-free options. Always check with the restaurant for accurate information regarding food allergens and dietary restrictions

**OpenAI:** We have several gluten-free burger options on our menu. However, it's important to note that currently, all the burgers listed as gluten-free are marked as unavailable. Here are the options:

Crunch Burger
Original Fillet Breaded
Fire Zinger Stacker
Fire Zinger
Colonel Stacker
Colonel Burger.
Please note that other burgers listed contain allergens such as wheat and soy, and hence, are not gluten-free. Would you like to pre-order one of the gluten-free options to be notified when they become available?

▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄

- "How many calories does the Colonel have?"

**MistralAI:** The question is ambiguous as it does not specify whether "the Colonel" refers to a specific food item or drink on the menu. However, assuming that it refers to a food item, there are two options on the menu with "Colonel" in their names: "Colonel Stacker" and "Colonel Burger". Based on the context information provided, the "Colonel Stacker" has 150 calories and the "Colonel Burger" also has 150 calories. However, it is important to note that all items on the menu are listed as not available, so it is unclear whether these items are currently offered or not. Additionally, the context information does not provide any information about the nutritional content of these items beyond the number of calories, so it is not possible to provide a more detailed answer without additional context.
 
**OpenAI:** The Colonel Burger has 150 calories, 100 grams of fat, and 10 grams of protein

▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄─▀▄

- "Can I get a Whopper?"

**MistralAI:** 
Based on the information provided, it appears that the menu does not include a "Whopper" burger. All available burger options and their details are listed under the "Burgers" category, and a "Whopper" is not among them. Therefore, unfortunately, you cannot get a Whopper from this menu.


**OpenAI:** I could not find a "Whopper" listed in the document. It appears that this specific item may not be available on our menu. If there's another burger or item you're interested in, feel free to ask!
