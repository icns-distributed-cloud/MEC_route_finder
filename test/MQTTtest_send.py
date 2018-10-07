from paho.mqtt import publish

print("publish - cart/room/starting_room_number")
publish.single("cart/room/starting_room_number", "3", hostname="163.180.117.195", port=1883)
# cart.publish("cart/room/starting_room_number", "3", 2)
print("publish - cart/room/destination_room_number")
publish.single("cart/room/destination_room_number", "4", hostname="163.180.117.195", port=1883)
# cart.publish("cart/room/destination_room_number", "4", 2)
print("publish - cart/parking")
publish.single("cart/parking", "0", hostname="163.180.117.195", port=1883)
# cart.publish("cart/parking", "parking", 2)


