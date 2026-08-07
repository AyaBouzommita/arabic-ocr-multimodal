from typing import List, Callable, Any

def sort_boxes_rtl(items: List[Any], get_bbox: Callable[[Any], List[float]], y_threshold: float = 15.0) -> List[Any]:
    """Sort bounding boxes into horizontal lines, and sort each line Right-to-Left.
    
    Args:
        items: List of items to sort (e.g. YOLO regions, OCR tokens).
        get_bbox: Function that returns [x_min, y_min, x_max, y_max] for an item.
        y_threshold: Maximum pixel difference in Y to be considered the same line.
        
    Returns:
        A new list of items sorted top-to-bottom, right-to-left.
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
        
    # 3. Sort each line Right-to-Left (decreasing x_min)
    final_sorted = []
    for line in lines:
        line.sort(key=lambda item: get_bbox(item)[0], reverse=True)
        final_sorted.extend(line)
        
    return final_sorted
