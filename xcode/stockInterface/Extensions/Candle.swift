//
//  Candle.swift
//  stockInterface
//
//  Created by CHONK on 9/29/23.
//

import CoreData

extension Candle {
    
    @discardableResult
    convenience init(volume: Int64,
                     volumeWeighted: Float,
                     timestamp: Date,
                     transactionCount: Int64,
                     open: Float,
                     close: Float,
                     high: Float,
                     low: Float,
                     context: NSManagedObjectContext = CoreDataStack.TempContext) {
        self.init(context: context)
        self.volume = volume
        self.volumeWeighted = volumeWeighted
        self.date = timestamp
        self.transactionCount = transactionCount
        self.open = open
        self.close = close
        self.high = high
        self.low = low
    }
}
