"""测试 Profiling Bus 通信。"""

from shadowinfer.core.bus import MESSAGE_TYPES, ProfilingBus
from shadowinfer.core.structs import Message


class TestProfilingBus:
    def test_subscribe_and_send(self):
        """验证订阅和点对点发送。"""
        bus = ProfilingBus(name="test")
        received = []

        def callback(msg):
            received.append(msg)

        bus.subscribe("agent_a", callback)
        msg = Message.create(
            source="orchestrator",
            target="agent_a",
            message_type=MESSAGE_TYPES.REQUEST,
            payload={"data": 42},
            step_id=0,
        )
        bus.send(msg)
        assert len(received) == 1
        assert received[0].payload["data"] == 42
        assert received[0].message_type == MESSAGE_TYPES.REQUEST

    def test_broadcast(self):
        """验证广播消息。"""
        bus = ProfilingBus(name="test")
        received_a = []
        received_b = []

        bus.subscribe("agent_a", lambda m: received_a.append(m))
        bus.subscribe("agent_b", lambda m: received_b.append(m))

        msg = Message.create(
            source="orchestrator",
            target="broadcast",
            message_type=MESSAGE_TYPES.BROADCAST,
            payload={"info": "hello"},
            step_id=1,
        )
        bus.broadcast(msg)
        assert len(received_a) == 1
        assert len(received_b) == 1
        assert received_a[0].payload["info"] == "hello"
        assert received_b[0].payload["info"] == "hello"

    def test_message_log(self):
        """验证消息日志记录。"""
        bus = ProfilingBus(name="test")
        bus.subscribe("agent", lambda m: None)

        msg1 = Message.create(
            source="src", target="agent", message_type=MESSAGE_TYPES.REQUEST, payload={}
        )
        msg2 = Message.create(
            source="src", target="agent", message_type=MESSAGE_TYPES.RESPONSE, payload={}
        )
        bus.send(msg1)
        bus.send(msg2)
        logs = bus.get_message_log()
        assert len(logs) == 2
        assert logs[0].message_type == MESSAGE_TYPES.REQUEST
        assert logs[1].message_type == MESSAGE_TYPES.RESPONSE

    def test_message_filtering(self):
        """验证消息日志过滤。"""
        bus = ProfilingBus(name="test")
        bus.subscribe("agent_a", lambda m: None)
        bus.subscribe("agent_b", lambda m: None)

        msg_a = Message.create(
            source="src", target="agent_a", message_type=MESSAGE_TYPES.REQUEST, payload={}
        )
        msg_b = Message.create(
            source="src", target="agent_b", message_type=MESSAGE_TYPES.RESPONSE, payload={}
        )
        bus.send(msg_a)
        bus.send(msg_b)

        filtered_target = bus.get_message_log(target="agent_a")
        assert len(filtered_target) == 1
        assert filtered_target[0].target == "agent_a"

        filtered_type = bus.get_message_log(message_type=MESSAGE_TYPES.RESPONSE)
        assert len(filtered_type) == 1
        assert filtered_type[0].message_type == MESSAGE_TYPES.RESPONSE

        filtered_source = bus.get_message_log(source="src")
        assert len(filtered_source) == 2

    def test_message_stats(self):
        """验证消息统计。"""
        bus = ProfilingBus(name="test")
        bus.subscribe("agent", lambda m: None)

        for _ in range(3):
            bus.send(
                Message.create(
                    source="src", target="agent", message_type=MESSAGE_TYPES.REQUEST, payload={}
                )
            )
        for _ in range(2):
            bus.send(
                Message.create(
                    source="src", target="agent", message_type=MESSAGE_TYPES.RESPONSE, payload={}
                )
            )

        stats = bus.get_message_stats()
        assert stats[MESSAGE_TYPES.REQUEST] == 3
        assert stats[MESSAGE_TYPES.RESPONSE] == 2

    def test_unsubscribe(self):
        """验证取消订阅后不再接收消息。"""
        bus = ProfilingBus(name="test")
        received = []
        bus.subscribe("agent", lambda m: received.append(m))
        bus.unsubscribe("agent")
        bus.send(
            Message.create(
                source="src", target="agent", message_type=MESSAGE_TYPES.REQUEST, payload={}
            )
        )
        assert len(received) == 0

    def test_subscriber_count(self):
        """验证订阅者计数。"""
        bus = ProfilingBus(name="test")
        assert bus.get_subscriber_count() == 0
        bus.subscribe("a", lambda m: None)
        bus.subscribe("b", lambda m: None)
        assert bus.get_subscriber_count() == 2

    def test_clear_log(self):
        """验证清空日志。"""
        bus = ProfilingBus(name="test")
        bus.subscribe("agent", lambda m: None)
        bus.send(
            Message.create(
                source="src", target="agent", message_type=MESSAGE_TYPES.REQUEST, payload={}
            )
        )
        bus.clear_log()
        assert len(bus.get_message_log()) == 0
