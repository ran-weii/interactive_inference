import numpy as np

def test_closest_point_on_line():
    from src.data.utils import closest_point_on_line
    x, y = 0, 0 
    
    # case 1: closest is tangent
    x_line1 = [-1, 1]
    y_line1 = [1, 1]
    a1, b1 = closest_point_on_line(x, y, x_line1, y_line1)
    assert a1 == 0 and b1 == 1
    
    # case 2: closest is left
    x_line2 = [1, 2]
    y_line2 = [1, 1]
    a2, b2 = closest_point_on_line(x, y, x_line2, y_line2)
    assert a2 == 1 and b2 == 1
    
    # case 3: closest is right
    x_line3 = [-2, -1]
    y_line3 = [1, 1]
    a3, b3 = closest_point_on_line(x, y, x_line3, y_line3)
    assert a3 == -1 and b3 == 1
    
    print("test_closest_point_on_line passed")

def test_get_cardinal_direction():
    from src.data.utils import get_cardinal_direction
    x, y = 0, 0
    heading = np.deg2rad(45)
    
    # case 1: front left
    heading_vec1 = np.deg2rad(60)
    card1 = get_cardinal_direction(x, y, heading, 1, np.tan(heading_vec1))
    card1_ = heading_vec1 - heading
    assert card1 == card1_
    assert card1 > 0 and card1 < np.pi # left
    assert card1 > -np.pi / 2 and card1 < np.pi / 2 # front
    
    # case 2: front right
    heading_vec2 = np.deg2rad(30)
    card2 = get_cardinal_direction(x, y, heading, 1, np.tan(heading_vec2))
    card2_ = heading_vec2 - heading
    assert card2 == card2_
    assert card2 < 0 and card2 > -np.pi # right
    assert card2 > -np.pi / 2 and card2 < np.pi / 2 # front
    
    # case 3: behind left
    heading_vec3 = np.deg2rad(30)
    card3 = get_cardinal_direction(x, y, heading, -1, np.tan(heading_vec3))
    card3_ = np.deg2rad(180) - heading_vec3 - heading
    assert card3 == card3_
    assert card3 > 0 and card3 < np.pi # left
    assert card3 > np.pi/2 or card3 < -np.pi/2 # behind
    
    # case 4: behind right
    heading_vec4 = np.deg2rad(-60)
    card4 = get_cardinal_direction(x, y, heading, 1, np.tan(heading_vec4))
    card4_ = heading_vec4 - heading
    assert card4 == card4_
    assert card4 < 0 and card4 > -np.pi # right
    assert card4 > np.pi/2 or card4 < -np.pi/2 # behind
    
    print("test_get_cardianl_direction passed")

def test_is_above_line():
    from src.data.utils import is_above_line
    x, y = 0, 0
    
    # case 0: on the line
    heading0 = np.tan(30)
    above0 = is_above_line(x, y, heading0, 0, 0)
    assert above0 == 0
    
    # case 1: positive slope, above
    heading1 = np.deg2rad(30)
    heading_vec1 = np.deg2rad(45)
    above1 = is_above_line(x, y, heading1, 1, np.tan(heading_vec1))
    assert above1 == 1
    
    # case 2: positve slope, below
    heading2 = np.deg2rad(30)
    heading_vec2 = np.deg2rad(20)
    above2 = is_above_line(x, y, heading2, 1, np.tan(heading_vec2))
    assert above2 == -1
    
    # case 3: negative slope, above
    heading3 = np.deg2rad(-30)
    heading_vec3 = np.deg2rad(-20)
    above3 = is_above_line(x, y, heading3, 1, np.tan(heading_vec3))
    assert above3 == 1
    
    # case 4: negative slope, below
    heading4 = np.deg2rad(-30)
    heading_vec4 = np.deg2rad(-60)
    above4 = is_above_line(x, y, heading4, 1, np.tan(heading_vec4))
    assert above4 == -1
    
    print("test_is_above_line passed")

if __name__ == "__main__":
    test_closest_point_on_line()
    test_get_cardinal_direction()
    test_is_above_line()