//
//  ChartManager.swift
//  stockInterface
//
//  Created by Jackson Tubbs on 11/12/23.
//

import Cocoa

class ChartManager {
    
    static let shared = ChartManager()
    
    weak var currentStockView: StockView?
    
    var currentAggregate: StockAggregate?
    var indicatorData: IndicatorData?
    
    func currentlyChartingStock() -> Bool {
        return currentAggregate != nil
    }
    
    func chartStock(_ aggregate: StockAggregate) {
        
        self.currentAggregate = aggregate
        currentStockView?.stockAggregate = aggregate
    }
    
    func setIndicatorData(_ indicatorData: IndicatorData) {
        self.indicatorData = indicatorData
        currentStockView?.addColoredLines(lines: [(indicatorData.ema14, CGColor.ChartEMALines.NSColor())])
        currentStockView?.addColoredLines(lines: [(indicatorData.ema28, CGColor.ChartEMALines.NSColor())])
        currentStockView?.addColoredLines(lines: [(indicatorData.sma50, CGColor.LightSMALine.NSColor())])
        currentStockView?.addColoredLines(lines: [(indicatorData.sma200, CGColor.DarkSMALine.NSColor())])
        currentStockView?.mouseDelegate = self
    }
    
    func setAuxSets(auxSets: [StockViewAuxGraphProperties]) {
        currentStockView?.setAuxViews(auxSets)
    }
}

extension ChartManager: StockViewMouseDelegate {
    func candleHoverChanged(index: Int) {
        guard let data = indicatorData else {return}
        guard let aggregate = currentAggregate else {return}
        if index < 0 || index >= data.length {return}
        
        LabelValueController.shared.setLabelValue(index: 0, label: "% Gain", value: "\((data.percentageChange[index] * 100).toRoundedString(precision: 2))%")
        LabelValueController.shared.setLabelValue(index: 1, label: "Close", value: "\((aggregate.candles[index].close).toRoundedString(precision: 2))")
        LabelValueController.shared.setLabelValue(index: 2, label: "% Deviation 5", value: "\((data.standardDeviationForPercentageChange1[index] * 100).toRoundedString(precision: 2))%")
        LabelValueController.shared.setLabelValue(index: 3, label: "Index", value: "\(index)")
        LabelValueController.shared.setLabelValue(index: 4, label: "V % Change", value: "\((data.volumeChange[index] * 100).toRoundedString(precision: 2))%")
        LabelValueController.shared.setLabelValue(index: 5, label: "% Deviation 30", value: "\((data.standardDeviationForPercentageChange2[index] * 100).toRoundedString(precision: 2))%")
        LabelValueController.shared.setLabelValue(index: 9, label: "AVG VOL", value: "\((data.averageVolume[index   ]).toRoundedString(precision: 2))%")

        LabelValueController.shared.setLabelValue(index: 12, label: "VOL * Price AVG", value: "\(data.volumeTCAverage[index].toRoundedString(precision: 1))")
        LabelValueController.shared.setLabelValue(index: 15, label: "Volume ", value: "\(Float(aggregate.candles[index].volume).toRoundedString(precision: 1))")
        
//        let date = aggregate.candles[index].timestamp
////        print("\n\n\(date)\n\(date.stripDateToDayMonthYear())")
////        date = date.stripDateToDayMonthYear()
//        let formatter = DateFormatter.AlpacaDateFormatter
//        LabelValueController.shared.setLabelValue(index: 6, label: "Old", value: formatter.string(from: date))
//        LabelValueController.shared.setLabelValue(index: 9, label: "New", value: formatter.string(from: date.stripDateToDayMonthYearAndAddOneDay()))
//        LabelValueController.shared.setLabelValue(index: 12, label: "SPY Date", value: formatter.string(from: SPYController.shared.spyAggregate.candles.first!.timestamp.stripDateToDayMonthYearAndAddOneDay()))
    }
}
