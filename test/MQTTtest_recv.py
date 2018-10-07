import paho.mqtt.client as mqtt

# ===================
# ============== Cart
def on_connect_cart(client, obj, flags, rc):
    if rc == 0:
        print("Cart connected with result code " + str(rc))
        cart.subscribe("cart/room/starting_room_number")
        cart.subscribe("cart/room/destination_room_number")
        cart.subscribe("cart/parking")

    else:
        print("Bad connection returned code = ", rc)


def on_message_cart(client, obj, msg):
    print("Cart new message: " + msg.topic + " " + str(msg.payload))
    if msg.topic == "cart/room/starting_room_number":
        floor = str(msg.payload)
        # add function
        print(floor)

    elif msg.topic == "cart/room/destination_room_number":
        floor = str(msg.payload)
        # add function
        print(floor)
    elif msg.topic == "cart/parking":
        print(str(msg.payload))
    else:
        print("Unknown topic")


def on_publish_cart(client, obj, mid):
    print("mid: " + str(mid))


def on_subscribe_cart(client, obj, mid, granted_qos):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))


def on_log_cart(client, obj, level, string):
    print(string)


cart = mqtt.Client("cart")
cart.on_connect = on_connect_cart
cart.on_message = on_message_cart

### Connect to MQTT broker
try:
    cart.connect("163.180.117.195", 1883, 60)
except:
    print("ERROR: Could not connect to MQTT")

    # The below lines will be used to publish the topics
    # cart.publish("elevator/starting_floor_number", "3", 2)
    # cart.publish("elevator/destination_floor_number", "4", 2)

cart.loop_forever()
