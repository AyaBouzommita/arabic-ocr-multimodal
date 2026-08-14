from typing import List, Callable, Any

def sort_boxes_smart(items: List[Any], get_bbox: Callable[[Any], List[float]], is_rtl: bool = False, y_threshold: float = 15.0) -> List[Any]:
    """Sort bounding boxes into horizontal lines, and sort each line appropriately.
    
    Args:
        items: List of items to sort (e.g. YOLO regions, OCR tokens).
        get_bbox: Function that returns [x_min, y_min, x_max, y_max] for an item.
        is_rtl: If True, sort Right-to-Left. If False, sort Left-to-Right.
        y_threshold: Maximum pixel difference in Y to be considered the same line.
        
    Returns:
        A new list of items sorted top-to-bottom, and then by reading direction.
    """
    if not items:
        return []
        
    # 1. Sort primarily top-to-bottom based on y_min
    sorted_by_y = sorted(items, key=lambda item: get_bbox(item)[1])
    
    # 2. Group into lines based on y_threshold
    lines = []
    current_line = []
    current_y = None
    
    for item in sorted_by_y:
        y_min = get_bbox(item)[1]
        
        if current_y is None:
            current_line.append(item)
            current_y = y_min
        else:
            if abs(y_min - current_y) <= y_threshold:
                current_line.append(item)
            else:
                lines.append(current_line)
                current_line = [item]
                current_y = y_min
                
    if current_line:
        lines.append(current_line)
        
    # 3. Sort each line Left-to-Right or Right-to-Left
    final_sorted = []
    for line in lines:
        if is_rtl:
            line.sort(key=lambda item: get_bbox(item)[0], reverse=True)
        else:
            line.sort(key=lambda item: get_bbox(item)[0], reverse=False)
        final_sorted.extend(line)
        
    return final_sorted
