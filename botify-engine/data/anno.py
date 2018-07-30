
from mongoengine import *
from bson import ObjectId

class EntityMention(EmbeddedDocument):
    id = ObjectIdField(db_field="_id", default=ObjectId())
    role = StringField(db_field = 'name')
    start = IntField()
    end = IntField()

class Question(Document):
    business_id = ObjectIdField()
    text = StringField()
    has_intent = BooleanField(db_field = 'has_action')
    intent_id = ObjectIdField(db_field = 'action_id')
    entity_mentions = ListField(EmbeddedDocumentField(EntityMention), db_field='entity_instances')
    upload_time = DateTimeField()
    annotated = BooleanField()

class Entity(Document):
    business_id = ObjectIdField()
    name = StringField(db_field = 'text')
    desc = StringField()

class Parameter(EmbeddedDocument):
    required = BooleanField(default=False)
    open_vocab = BooleanField(default=False)
    eid = ObjectIdField()
    entity = StringField()
    name = StringField()
    prompts = ListField(StringField())

class Intent(Document):
    business_id = ObjectIdField()
    priority = IntField()
    name = StringField(db_field = 'text')
    desc = StringField()
    roles = DynamicField(db_field = 'schemas', default = {}) # a dict that maps role to entity object id
    params = ListField(EmbeddedDocumentField(Parameter))
    prompts = ListField(StringField())
    examples = ListField(StringField())
    suppressed = BooleanField(default=False)

class Business(Document):
    email = EmailField()
    name = StringField()

def get_entity(e_mention, intent):
    """e_mention: EntityMention,
    intent: Intent
    """

    return Entity.objects(id = intent.roles[e_mention.role]).first()
