
import pytest

from Otolits_identyfication_program.bounding_box import BoundingBox
from Otolits_identyfication_program.bounding_box_manager import BoundingBoxManager

def test_add_box():
    manager = BoundingBoxManager()
    manager.add_box(10, 20, 50, 60)
    assert len(manager.get_boxes()) == 1

def test_remove_box():
    manager = BoundingBoxManager()
    box = BoundingBox(10, 20, 50, 60)
    manager.add_box(10, 20, 50, 60)
    manager.remove_box(box)
    assert len(manager.get_boxes()) == 0

def test_update_box():
    manager = BoundingBoxManager()
    box = BoundingBox(10, 20, 50, 60)
    manager.add_box(10, 20, 50, 60)
    manager.update_box(box, 15, 25, 55, 65)
    updated_box = manager.get_boxes()[0]
    assert updated_box.x1 == 15
    assert updated_box.y1 == 25
    assert updated_box.x2 == 55
    assert updated_box.y2 == 65

def test_get_box_at():
    manager = BoundingBoxManager()
    manager.add_box(10, 20, 50, 60)
    found_box = manager.get_box_at(30, 40)
    assert found_box is not None
    assert found_box.x1 == 10
    assert found_box.y1 == 20

def test_get_box_at_no_match():
    manager = BoundingBoxManager()
    manager.add_box(10, 20, 50, 60)
    found_box = manager.get_box_at(100, 100)
    assert found_box is None