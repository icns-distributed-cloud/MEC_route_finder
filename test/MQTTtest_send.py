from paho.mqtt import publish

print("publish - cart/room/starting_room_number")
publish.single("cart/room/starting_room_number", "330", hostname="163.180.117.195", port=1883)

print("publish - cart/room/destination_room_number")
publish.single("cart/room/destination_room_number", "323-1", hostname="163.180.117.195", port=1883)

print("publish - cart/room/destination_room_number")
publish.single("cart/room/destination_room_number", "250", hostname="163.180.117.195", port=1883)

print("publish - cart/parking")
publish.single("cart/parking", "0", hostname="163.180.117.195", port=1883)


