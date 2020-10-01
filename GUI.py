import tkinter as tk
from PIL import Image, ImageTk

def suggestionPage():
    top3 = tk.Toplevel()
    top3.title("Suggestion")
    top3.geometry("400x400")
    l = tk.Label(top3, text="BUY on May 5. Sell on May 17", bg="red",font=30)
    l.pack()

def newsAttitudePage():
    top2 = tk.Toplevel()
    top2.title("Input news attitude")
    top2.geometry("400x400")
    text1 = tk.StringVar()
    text2 = tk.StringVar()
    text3 = tk.StringVar()
    text4 = tk.StringVar()
    submitButton = tk.Button(top2, text="Submit", font=20, command=suggestionPage)
    technologyLabel = tk.Label(top2, bg='green', font=16, text="Technology news attitude")
    agricultureLabel = tk.Label(top2, bg='green', font=16, text="Agriculture news attitude")
    entertainmentLabel = tk.Label(top2, bg='green', font=16, text="Entertainment news attitude")
    finantialLabel = tk.Label(top2, bg='green', font=16, text="Finantial news attitude")
    inputForm1 = tk.Entry(top2, textvariable=text1)
    inputForm2 = tk.Entry(top2, textvariable=text2)
    inputForm3 = tk.Entry(top2, textvariable=text3)
    inputForm4 = tk.Entry(top2, textvariable=text4)
    text1.set("")
    text2.set("")
    text3.set("")
    text4.set("")
    technologyLabel.pack()
    inputForm1.pack()
    agricultureLabel.pack()
    inputForm2.pack()
    entertainmentLabel.pack()
    inputForm3.pack()
    finantialLabel.pack()
    inputForm4.pack()
    technologyText = text1.get()
    agricultureText = text1.get()
    entertainmentText = text1.get()
    finantialText = text1.get()
    submitButton.pack()

    top2.mainloop()

def selectPage():
    top1 = tk.Toplevel()
    top1.title('Select a stock')
    top1.geometry('400x400')
    confirmButton = tk.Button(top1, text="Confirm", command=newsAttitudePage, font=20)
    confirmButton.pack()
    var = tk.StringVar()
    l = tk.Label(top1, bg='yellow', width=20)
    l.pack()
    def stock1():
        l.config(text='DIS',font=("Purisa", 20))
    def stock2():
        l.config(text='GOOG',font=("Purisa", 20))
    def stock3():
        l.config(text='HD',font=("Purisa", 20))
    def stock4():
        l.config(text='JNJ',font=("Purisa", 20))
    def stock5():
        l.config(text='KHC',font=("Purisa", 20))
    def stock6():
        l.config(text='MDLZ',font=("Purisa", 20))
    def stock7():
        l.config(text='MO',font=("Purisa", 20))
    def stock8():
        l.config(text='NFLX',font=("Purisa", 20))
    def stock9():
        l.config(text='PM',font=("Purisa", 20))
    def stock10():
        l.config(text='TGT',font=("Purisa", 20))

    r1 = tk.Radiobutton(top1, text='Stock No.1', font=("Purisa", 20), variable=var, value='000001', command=stock1)
    r1.pack()
    r2 = tk.Radiobutton(top1, text='Stock No.2', font=("Purisa", 20), variable=var, value='000002', command=stock2)
    r2.pack()
    r3 = tk.Radiobutton(top1, text='Stock No.3', font=("Purisa", 20), variable=var, value='000003', command=stock3)
    r3.pack()
    r4 = tk.Radiobutton(top1, text='Stock No.4', font=("Purisa", 20), variable=var, value='000004', command=stock4)
    r4.pack()
    r5 = tk.Radiobutton(top1, text='Stock No.5', font=("Purisa", 20), variable=var, value='000005', command=stock5)
    r5.pack()
    r6 = tk.Radiobutton(top1, text='Stock No.6', font=("Purisa", 20), variable=var, value='000006', command=stock6)
    r6.pack()
    r7 = tk.Radiobutton(top1, text='Stock No.7', font=("Purisa", 20), variable=var, value='000007', command=stock7)
    r7.pack()
    r8 = tk.Radiobutton(top1, text='Stock No.8', font=("Purisa", 20), variable=var, value='000008', command=stock8)
    r8.pack()
    r9 = tk.Radiobutton(top1, text='Stock No.9', font=("Purisa", 20), variable=var, value='000009', command=stock9)
    r9.pack()
    r10 = tk.Radiobutton(top1, text='Stock No.10', font=("Purisa", 20), variable=var, value='000010', command=stock10)
    r10.pack()

    top1.mainloop()


window=tk.Tk()
canvas = tk.Canvas(window, width=1000, height=600)
homePageBG = Image.open("HomePageBackGround.png")
homePageBG = homePageBG.resize((1000,600))
photo = ImageTk.PhotoImage(homePageBG)
canvas.create_image(500,300,image=photo)
canvas.create_text(800,550, text = 'Designed by Group A', fill='red', font=("Purisa", 20))
canvas.create_text(500,100, text = 'Billionaire', fill='red', font=("magik", 40))



button = tk.Button(window, text="Begin", command=selectPage, font=12)
canvas_widget = canvas.create_window(500,450,window=button, width=100, height=40)
canvas.pack()
window.mainloop()
