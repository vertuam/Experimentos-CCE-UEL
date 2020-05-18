from entities import *
from entities.basic import PRIMARY, SECONDARY
from entities.entity import Entity
from entities.field import IntegerField, StringField


class Case(Entity):
    case_id = IntegerField(group=PRIMARY)
    event_position = IntegerField(group=SECONDARY)
    activity_name = StringField(group=SECONDARY)
    timestamp = StringField(group=SECONDARY)
    user = StringField(group=SECONDARY)
    label = StringField(group=SECONDARY)
    anomaly_type = StringField(group=SECONDARY)
    anomaly_description = StringField(group=SECONDARY)
