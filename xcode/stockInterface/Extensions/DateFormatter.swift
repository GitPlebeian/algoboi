//
//  DateFormatter.swift
//  stockInterface
//
//  Created by CHONK on 9/29/23.
//

import Foundation

extension DateFormatter {
    
    static var AlpacaDateFormatter: DateFormatter = {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ssZ"
        return dateFormatter
    }()
}
