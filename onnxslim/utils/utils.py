def format_bytes(size_in_bytes):
    # Define the units and their corresponding suffixes
    units = ['B', 'KB', 'MB', 'GB']
    
    # Determine the appropriate unit and conversion factor
    unit_index = 0
    while size_in_bytes >= 1024 and unit_index < len(units) - 1:
        size_in_bytes /= 1024
        unit_index += 1
    
    # Format the result with two decimal places
    formatted_size = "{:.2f} {}".format(size_in_bytes, units[unit_index])
    return formatted_size
