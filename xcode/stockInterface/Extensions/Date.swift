//
//  Date.swift
//  stockInterface
//
//  Created by CHONK on 2/7/24.
//

import Foundation

extension Date {
    
    func stripDateToDayMonthYearAndAddOneDay() -> Date {
        let calendar = Calendar.current
        var components = calendar.dateComponents([.year, .month, .day], from: self)
        components.day = (components.day ?? 0) + 1 // Increment day by 1
        return calendar.date(from: components)!
    }
}
