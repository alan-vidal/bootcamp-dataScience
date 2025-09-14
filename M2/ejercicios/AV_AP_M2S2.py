def main():
    precioProducto = int(input("Precio Producto: $"))
    cantidad = int(input("Cantidad de Producto:"))
    descuento = int(input("Descuento (%):"))

    totalSinDescuento = precioProducto * cantidad
    montoDescuento = float(totalSinDescuento * (descuento/100))
    totalConDescuento = totalSinDescuento - montoDescuento

    print ("----------------")
    print("Total sin descuento: $", totalSinDescuento)
    print("Monto de descuento: $", montoDescuento)
    print("Total con descuento: $", totalConDescuento)

if __name__=="__main__":
    main()
